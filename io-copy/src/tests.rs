use std::{
    fs::OpenOptions,
    io::{BufRead, Read, Result, Seek, SeekFrom, Write},
};

#[test]
fn copy_specialization() -> Result<()> {
    use std::io::{BufReader, BufWriter};

    let tmp_path = tempfile::tempdir().unwrap();
    let source_path = tmp_path.path().join("copy-spec.source");
    let sink_path = tmp_path.path().join("copy-spec.sink");

    let mut source = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&source_path)
        .unwrap();
    source.write_all(b"abcdefghiklmnopqr").unwrap();
    source.seek(SeekFrom::Start(8)).unwrap();
    let mut source = BufReader::with_capacity(8, source.take(5));
    source.fill_buf().unwrap();
    assert_eq!(source.buffer(), b"iklmn");
    source.get_mut().set_limit(6);
    source.get_mut().get_mut().seek(SeekFrom::Start(1)).unwrap(); // "bcdefg"
    let mut source = source.take(10); // "iklmnbcdef"

    let mut sink = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&sink_path)
        .unwrap();
    sink.write_all(b"000000").unwrap();
    let mut sink = BufWriter::with_capacity(5, sink);
    sink.write_all(b"wxyz").unwrap();
    assert_eq!(sink.buffer(), b"wxyz");

    let copied = crate::copy(&mut source, &mut sink).unwrap();
    assert_eq!(copied, 10, "copy obeyed limit imposed by Take");
    assert_eq!(sink.buffer().len(), 0, "sink buffer was flushed");
    assert_eq!(source.limit(), 0, "outer Take was exhausted");
    assert_eq!(
        source.get_ref().buffer().len(),
        0,
        "source buffer should be drained"
    );
    assert_eq!(
        source.get_ref().get_ref().limit(),
        1,
        "inner Take allowed reading beyond end of file, some bytes should be left"
    );

    let mut sink = sink.into_inner().unwrap();
    sink.seek(SeekFrom::Start(0)).unwrap();
    let mut copied = Vec::new();
    sink.read_to_end(&mut copied).unwrap();
    assert_eq!(&copied, b"000000wxyziklmnbcdef");

    let rm1 = std::fs::remove_file(source_path);
    let rm2 = std::fs::remove_file(sink_path);

    rm1.and(rm2)
}

#[test]
fn copies_append_mode_sink() -> Result<()> {
    let tmp_path = tempfile::tempdir().unwrap();
    let source_path = tmp_path.path().join("copies_append_mode.source");
    let sink_path = tmp_path.path().join("copies_append_mode.sink");
    let mut source = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(&source_path)?;
    write!(source, "not empty")?;
    source.seek(SeekFrom::Start(0))?;
    let mut sink = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&sink_path)?;

    let copied = crate::copy(&mut source, &mut sink)?;

    assert_eq!(copied, 9);

    Ok(())
}
