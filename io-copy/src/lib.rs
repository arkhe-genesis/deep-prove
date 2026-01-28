#![allow(internal_features)]
#![feature(rustc_attrs, min_specialization)]

#[cfg(test)]
mod tests;

use std::io::{Read, Result, Write};

/// Copies the entire contents of a reader into a writer.
///
/// This works almost the same as `std::io::copy` except that `safe_kernel_copy`
/// check has been removed to allow copy from a file to a socket in kernel space
/// without the overhead of the file being copied into the user space.
///
/// The caller should ensure that the source file cannot be modified for at
/// least the duration of its transfer.
pub fn copy<R: Read + ?Sized, W: Write + ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64> {
    copy_impl::copy_spec(reader, writer)
}

mod copy_impl {
    //! Based on Rust `library/std/src/sys/io/kernel_copy/linux.rs` with some
    //! modifications. The main thing we care about is sendfile from FD to a TCP
    //! stream and vice-versa.

    use std::{
        cmp::min,
        fs::{File, Metadata},
        io::{Error, Read, Result, Write},
        mem::ManuallyDrop,
        net::TcpStream,
        os::{
            fd::{AsRawFd, FromRawFd, RawFd},
            unix::fs::FileTypeExt,
        },
        ptr,
        sync::atomic::{AtomicBool, Ordering},
    };

    #[cfg(target_os = "macos")]
    use libc::sendfile as sendfile_macos;
    #[cfg(not(target_os = "macos"))]
    use libc::sendfile64;
    use libc::{EINVAL, ENOSYS, EOVERFLOW, EPERM};

    pub(crate) fn copy_spec<R: Read + ?Sized, W: Write + ?Sized>(
        read: &mut R,
        write: &mut W,
    ) -> Result<u64> {
        let copier = Copier { read, write };
        SpecCopy::copy(copier)
    }

    struct Copier<'a, 'b, R: Read + ?Sized, W: Write + ?Sized> {
        read: &'a mut R,
        write: &'b mut W,
    }

    trait SpecCopy {
        fn copy(self) -> Result<u64>;
    }

    impl<R: Read + ?Sized, W: Write + ?Sized> SpecCopy for Copier<'_, '_, R, W> {
        default fn copy(self) -> Result<u64> {
            std::io::copy(self.read, self.write)
        }
    }

    impl<R: CopyRead, W: CopyWrite> SpecCopy for Copier<'_, '_, R, W> {
        fn copy(self) -> Result<u64> {
            let (reader, writer) = (self.read, self.write);
            let r_cfg = reader.properties();
            let w_cfg = writer.properties();

            // before direct operations on file descriptors ensure that all source and sink buffers are empty
            let mut flush = || -> Result<u64> {
                let bytes = reader.drain_to(writer, u64::MAX)?;
                // BufWriter buffered bytes have already been accounted for in earlier write() calls
                writer.flush()?;
                Ok(bytes)
            };

            let mut written = 0u64;

            if let (CopyParams(input_meta, Some(readfd)), CopyParams(_output_meta, Some(writefd))) =
                (r_cfg, w_cfg)
            {
                written += flush()?;
                let max_write = reader.min_limit();

                if input_meta.potential_sendfile_source() {
                    let result = sendfile(readfd, writefd, max_write);
                    result.update_take(reader);

                    match result {
                        CopyResult::Ended(bytes_copied) => return Ok(bytes_copied + written),
                        CopyResult::Error(e, _) => return Err(e),
                        CopyResult::Fallback(bytes) => written += bytes,
                    }
                }
            }

            // fallback if none of the more specialized syscalls wants to work with these file descriptors
            match std::io::copy(reader, writer) {
                Ok(bytes) => Ok(bytes + written),
                err => err,
            }
        }
    }

    struct CopyParams(FdMeta, Option<RawFd>);

    /// This type represents either the inferred `FileType` of a `RawFd` based on the source
    /// type from which it was extracted or the actual metadata
    ///
    /// The methods on this type only provide hints, due to `AsRawFd` and `FromRawFd` the inferred
    /// type may be wrong.
    enum FdMeta {
        Metadata(Metadata),
        Socket,
        /// We don't have any metadata because the stat syscall failed
        NoneObtained,
    }

    impl FdMeta {
        fn potential_sendfile_source(&self) -> bool {
            match self {
                // procfs erroneously shows 0 length on non-empty readable files.
                // and if a file is truly empty then a `read` syscall will determine that and skip the write syscall
                // thus there would be benefit from attempting sendfile
                FdMeta::Metadata(meta)
                    if meta.file_type().is_file() && meta.len() > 0
                        || meta.file_type().is_block_device() =>
                {
                    true
                }
                _ => false,
            }
        }
    }

    #[rustc_specialization_trait]
    trait CopyRead: Read {
        /// Implementations that contain buffers (i.e. `BufReader`) must transfer data from their internal
        /// buffers into `writer` until either the buffers are emptied or `limit` bytes have been
        /// transferred, whichever occurs sooner.
        /// If nested buffers are present the outer buffers must be drained first.
        ///
        /// This is necessary to directly bypass the wrapper types while preserving the data order
        /// when operating directly on the underlying file descriptors.
        fn drain_to<W: Write>(&mut self, _writer: &mut W, _limit: u64) -> Result<u64> {
            Ok(0)
        }

        /// Updates `Take` wrappers to remove the number of bytes copied.
        fn taken(&mut self, _bytes: u64) {}

        /// The minimum of the limit of all `Take<_>` wrappers, `u64::MAX` otherwise.
        /// This method does not account for data `BufReader` buffers and would underreport
        /// the limit of a `Take<BufReader<Take<_>>>` type. Thus its result is only valid
        /// after draining the buffers via `drain_to`.
        fn min_limit(&self) -> u64 {
            u64::MAX
        }

        /// Extracts the file descriptor and hints/metadata, delegating through wrappers if necessary.
        fn properties(&self) -> CopyParams;
    }

    #[rustc_specialization_trait]
    trait CopyWrite: Write {
        /// Extracts the file descriptor and hints/metadata, delegating through wrappers if necessary.
        fn properties(&self) -> CopyParams;
    }

    impl<T> CopyRead for &mut T
    where
        T: CopyRead,
    {
        fn drain_to<W: Write>(&mut self, writer: &mut W, limit: u64) -> Result<u64> {
            (**self).drain_to(writer, limit)
        }

        fn taken(&mut self, bytes: u64) {
            (**self).taken(bytes);
        }

        fn min_limit(&self) -> u64 {
            (**self).min_limit()
        }

        fn properties(&self) -> CopyParams {
            (**self).properties()
        }
    }

    impl<T> CopyWrite for &mut T
    where
        T: CopyWrite,
    {
        fn properties(&self) -> CopyParams {
            (**self).properties()
        }
    }

    impl CopyRead for File {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
        }
    }

    impl CopyRead for &File {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(*self), Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for File {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(self), Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for &File {
        fn properties(&self) -> CopyParams {
            CopyParams(fd_to_meta(*self), Some(self.as_raw_fd()))
        }
    }

    impl CopyRead for TcpStream {
        fn properties(&self) -> CopyParams {
            // avoid the stat syscall since we can be fairly sure it's a socket
            CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
        }
    }

    impl CopyRead for &TcpStream {
        fn properties(&self) -> CopyParams {
            // avoid the stat syscall since we can be fairly sure it's a socket
            CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for TcpStream {
        fn properties(&self) -> CopyParams {
            // avoid the stat syscall since we can be fairly sure it's a socket
            CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
        }
    }

    impl CopyWrite for &TcpStream {
        fn properties(&self) -> CopyParams {
            // avoid the stat syscall since we can be fairly sure it's a socket
            CopyParams(FdMeta::Socket, Some(self.as_raw_fd()))
        }
    }

    fn fd_to_meta<T: AsRawFd>(fd: &T) -> FdMeta {
        let fd = fd.as_raw_fd();
        let file: ManuallyDrop<File> = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
        match file.metadata() {
            Ok(meta) => FdMeta::Metadata(meta),
            Err(_) => FdMeta::NoneObtained,
        }
    }

    pub(super) enum CopyResult {
        Ended(u64),
        Error(Error, u64),
        Fallback(u64),
    }

    impl CopyResult {
        fn update_take(&self, reader: &mut impl CopyRead) {
            match *self {
                CopyResult::Fallback(bytes)
                | CopyResult::Ended(bytes)
                | CopyResult::Error(_, bytes) => reader.taken(bytes),
            }
        }
    }

    /// performs splice or sendfile between file descriptors
    /// Does _not_ fall back to a generic copy loop.
    fn sendfile(reader: RawFd, writer: RawFd, len: u64) -> CopyResult {
        static HAS_SENDFILE: AtomicBool = AtomicBool::new(true);

        if !HAS_SENDFILE.load(Ordering::Relaxed) {
            return CopyResult::Fallback(0);
        }

        let mut written = 0u64;
        while written < len {
            // according to its manpage that's the maximum size sendfile() will copy per invocation
            let chunk_size = min(len - written, 0x7ffff000_u64) as usize;

            let result = {
                #[cfg(target_os = "macos")]
                {
                    let mut written: i64 = chunk_size as i64;
                    let result = cvt(unsafe {
                        sendfile_macos(writer, reader, 0, &mut written, ptr::null_mut(), 0)
                    });
                    result.map(|_| written)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    cvt(unsafe { sendfile64(writer, reader, ptr::null_mut(), chunk_size) })
                }
            };

            match result {
                Ok(0) => break, // EOF
                Ok(ret) => written += ret as u64,
                Err(err) => {
                    return match err.raw_os_error() {
                        Some(ENOSYS | EPERM) => {
                            // syscall not supported (ENOSYS)
                            // syscall is disallowed, e.g. by seccomp (EPERM)
                            HAS_SENDFILE.store(false, Ordering::Relaxed);
                            assert_eq!(written, 0);
                            CopyResult::Fallback(0)
                        }
                        Some(EINVAL) => {
                            // splice/sendfile do not support this particular file descriptor (EINVAL)
                            assert_eq!(written, 0);
                            CopyResult::Fallback(0)
                        }
                        Some(os_err) if os_err == EOVERFLOW => CopyResult::Fallback(written),
                        _ => CopyResult::Error(err, written),
                    };
                }
            }
        }
        CopyResult::Ended(written)
    }

    // NOTE(Tomas): `cvt` is borrowed from `std::sys::pal::unix`
    trait IsMinusOne {
        fn is_minus_one(&self) -> bool;
    }

    macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

    impl_is_minus_one! { i8 i16 i32 i64 isize }

    /// Converts native return values to Result using the *-1 means error is in `errno`*  convention.
    /// Non-error values are `Ok`-wrapped.
    fn cvt<T: IsMinusOne>(t: T) -> Result<T> {
        if t.is_minus_one() {
            Err(Error::last_os_error())
        } else {
            Ok(t)
        }
    }
}
