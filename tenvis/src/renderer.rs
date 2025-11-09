use crate::Term;
use anyhow::{Context, bail, ensure};
use colored::Colorize;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;
use zkml::{Number, Shape, Tensor};

fn with_gnuplot(
    f: impl Fn(&mut dyn std::io::Write) -> anyhow::Result<Option<NamedTempFile>>,
) -> anyhow::Result<()> {
    let mut gnuplot_process = Command::new("gnuplot")
        .arg("-p")
        .stdout(Stdio::inherit())
        .stdin(Stdio::piped())
        .spawn()
        .context("starting gnuplot")?;

    let mut gnuplot = gnuplot_process
        .stdin
        .take()
        .context("opening stdin on gnuplot process")?;

    let out = f(&mut gnuplot)?;

    // Drop gnuplot STDIN so that it will exit
    drop(gnuplot);

    if let Some(temp_file) = out {
        gnuplot_process.wait().unwrap();
        open::that_in_background(temp_file.into_temp_path());
    }

    Ok(())
}

/// A Gnuplot renderer for a 1- or 2-D tensor.
pub(crate) struct GpTensor {
    /// The output device to use.
    pub term: Term,
    /// The title of the plot.
    pub title: String,
}
impl GpTensor {
    fn send_to_gnuplot<T: std::fmt::Display>(
        self,
        gp: &mut dyn std::io::Write,
        t: &Tensor<T>,
        p: &Projection,
    ) -> anyhow::Result<Option<NamedTempFile>> {
        let gp_outfile = match self.term {
            Term::Tty => {
                writeln!(gp, "set term dumb")?;
                None
            }
            Term::Qt => {
                writeln!(gp, "set term qt")?;
                None
            }
            Term::Sixel => {
                writeln!(gp, "set term sixelgd")?;
                None
            }
            Term::Png => {
                let out_file = tempfile::Builder::new()
                    .suffix(".png")
                    .disable_cleanup(true)
                    .tempfile()
                    .context("failed to create a temporary file")?;
                writeln!(gp, "set term png size 1000, 1000")?;
                writeln!(gp, "set output \"{}\"", out_file.path().display())?;
                Some(out_file)
            }
        };

        writeln!(
            gp,
            r#"
set title "{}"

set style function pm3d
set palette rgb 7,5,15

unset ytics
set autoscale xfix
set autoscale yfix
set autoscale cbfix

plot '-' matrix with image notitle

# For 3D-mesh visualization
# set mouse
# set dgrid3d 30,30
# splot '-' using 2:1:3 with lines
"#,
            self.title
        )?;

        if t.shape().rank() == 1 {
            // Hack, a 1-line only matrix would not print
            let mut ys = (0..2).peekable();
            while let Some(y) = ys.next() {
                let mut xs = (0..*(t.shape().get(p.x).unwrap())).peekable();
                while let Some(x) = xs.next() {
                    let v = p.at(t, x, y);
                    write!(gp, "{v}").unwrap();
                    if xs.peek().is_some() {
                        gp.write_all(" ".as_bytes()).unwrap();
                    }
                }
                if ys.peek().is_some() {
                    gp.write_all("\n".as_bytes())?;
                }
            }
        } else {
            let mut ys = (0..(t.shape().get(p.y).copied().unwrap_or(2))).peekable();
            while let Some(y) = ys.next() {
                let mut xs = (0..*(t.shape().get(p.x).unwrap())).peekable();
                while let Some(x) = xs.next() {
                    let v = p.at(t, x, y);
                    write!(gp, "{v}").unwrap();
                    if xs.peek().is_some() {
                        gp.write_all(" ".as_bytes()).unwrap();
                    }
                }
                if ys.peek().is_some() {
                    gp.write_all("\n".as_bytes())?;
                }
            }
        }

        Ok(gp_outfile)
    }
}

#[derive(Clone)]
pub(crate) struct Projection {
    shape: Shape,
    x: usize,
    y: usize,
    fixed: Vec<usize>,
    skips: Vec<usize>,
}
impl Projection {
    pub fn dim_to_char(dim: usize) -> anyhow::Result<char> {
        char::try_from('i' as u32 + dim as u32).with_context(|| format!("converting {dim} to char"))
    }

    pub fn new(shape: Shape) -> Projection {
        Projection {
            x: 0,
            y: 1,
            fixed: vec![0; shape.rank()],
            skips: Self::compute_skips(&shape),
            shape,
        }
    }

    pub fn set_from_string(&mut self, input: &str) -> anyhow::Result<()> {
        let input_dims = input.split(",").collect::<Vec<_>>();
        ensure!(
            input_dims.len() == self.shape.rank(),
            "input provided {} dims, but {} expected",
            input_dims.len(),
            self.shape.rank()
        );

        for (dim, input) in (0..self.shape.rank()).zip(input_dims.into_iter()) {
            if let Ok(fixed) = input.parse::<usize>() {
                ensure!(
                    fixed < self.shape[dim],
                    "unable to fix dimension {dim} to {fixed}: max value is {}",
                    self.shape[dim]
                );
                self.fixed[dim] = fixed;
            } else if input.to_lowercase() == "x" {
                self.x = dim;
            } else if input.to_lowercase() == "y" {
                self.y = dim;
            } else {
                bail!("unknown dimension projection: {input}");
            }
        }
        Ok(())
    }

    pub fn pretty(&self) -> String {
        (0..self.shape.rank())
            .map(|d| {
                if d == self.x {
                    "X".to_string()
                } else if d == self.y {
                    "Y".to_string()
                } else {
                    self.fixed[d].to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn compute_skips(shape: &Shape) -> Vec<usize> {
        (0..shape.len())
            .map(|k| shape.iter().skip(k + 1).product())
            .collect::<Vec<_>>()
    }

    fn idx_lst(&self, xi: usize, yi: usize) -> impl Iterator<Item = usize> + use<'_> {
        (0..self.shape.len()).map(move |dim| {
            if dim == self.x {
                xi
            } else if dim == self.y {
                yi
            } else {
                self.fixed[dim]
            }
        })
    }

    fn at<'a, T>(&'a self, t: &'a Tensor<T>, i: usize, j: usize) -> &'a T {
        &t.data()[self
            .idx_lst(i, j)
            .zip(self.skips.iter())
            .map(|(k, skip)| k * skip)
            .sum::<usize>()]
    }
}

struct BarcodeViewer<'t, T> {
    title: &'t str,
    /// A reference to the tensor to visualize.
    t: &'t Tensor<T>,
    /// How to project the tensor
    p: &'t Projection,
    /// The min. value in the tensor.
    min: T,
    /// The max. value in the tensor.
    max: T,
}
impl<'t, T: Number> BarcodeViewer<'t, T> {
    fn new(t: &'t Tensor<T>, p: &'t Projection, min: T, max: T, title: &'t str) -> Self {
        Self {
            title,
            t,
            p,
            min,
            max,
        }
    }
    fn render(&self, term: Term) -> anyhow::Result<()> {
        println!("{}", self.title);
        println!(
            "global tensor: min = {}, max = {}\n",
            format!("{}", self.min).bright_red(),
            format!("{}", self.max).bright_green()
        );

        match term {
            Term::Tty => {
                if self.t.shape().len() == 1 {
                    let (w, _) = term_size::dimensions().unwrap();
                    let per_char =
                        1.max((self.t.data().len() as f32 / ((w - 2) as f32)).ceil() as usize);
                    print!("┤");

                    for chunk in self.t.data().chunks(per_char) {
                        let sum = chunk.iter().fold(T::zero(), |ax, &x| ax + (x - self.min));
                        let avg = sum.to_f32().unwrap() / (chunk.len() as f32);
                        let projected =
                            ((avg / (self.max - self.min).to_f32().unwrap()) * 255.0) as u8;
                        print!("{}", " ".on_truecolor(projected, projected, projected));
                    }
                    println!("├");
                    println!("({per_char} elts. per char.)\n");

                    println!();
                    Ok(())
                } else {
                    let (w, _) = term_size::dimensions().unwrap();
                    let xlen = self.t.shape()[self.p.x];
                    let ylen = self.t.shape()[self.p.y];
                    let ratio = xlen as f32 / ylen as f32;

                    let xsize = xlen.min(w);
                    let ysize = (xlen as f32 / ratio) as usize;

                    let x_per_char = (xlen as f32 / xsize as f32) as usize;
                    let y_per_char = (ylen as f32 / ysize as f32) as usize;

                    let mut local_min = T::MAX;
                    let mut local_max = T::MIN;
                    println!("{}", "─".repeat(xsize));
                    for yi in (0..self.t.shape()[self.p.y]).step_by(y_per_char) {
                        let yrange = yi..(yi + y_per_char).min(self.t.shape()[self.p.y]);
                        for xi in (0..self.t.shape()[self.p.x]).step_by(x_per_char) {
                            let xrange = xi..(xi + x_per_char).min(self.t.shape()[self.p.x]);
                            let chunk_max = yrange
                                .clone()
                                .flat_map(|j| xrange.clone().map(move |i| self.p.at(self.t, i, j)))
                                .fold(T::MIN, |ax, x| ax.cmp_max(x));
                            let chunk_min = yrange
                                .clone()
                                .flat_map(|j| xrange.clone().map(move |i| self.p.at(self.t, i, j)))
                                .fold(T::MAX, |ax, x| ax.cmp_min(x));

                            local_min = local_min.cmp_min(&chunk_min);
                            local_max = local_max.cmp_max(&chunk_max);

                            let projected = (((chunk_max - self.min).to_f32().unwrap()
                                / (self.max - self.min).to_f32().unwrap())
                                * 255.0) as u8;

                            print!("{}", " ".on_truecolor(projected, projected, projected));
                        }
                        println!();
                    }
                    println!("{}", "─".repeat(xsize));
                    println!("X: {x_per_char} elts. per char., Y: {y_per_char} elts. per char.");
                    println!(
                        "projection min: {}, projection max: {}",
                        local_min.to_string().red(),
                        local_max.to_string().green()
                    );

                    Ok(())
                }
            }
            Term::Qt | Term::Sixel | Term::Png => with_gnuplot(|gp| {
                let t1d = GpTensor {
                    term,
                    title: format!(
                        "{} - glob. min.: {:.3}, glob. max.: {:.3}",
                        self.title, self.min, self.max
                    )
                    .clone(),
                };
                t1d.send_to_gnuplot(gp, self.t, self.p)
            }),
        }
    }
}

pub(crate) struct TensorViewer<'t, T> {
    /// A reference to the tensor to visualize.
    t: &'t Tensor<T>,
    /// How to project the tensor
    p: Projection,
    /// The min. value in the tensor.
    min: T,
    /// The max. value in the tensor.
    max: T,
}
impl<'t, T: Number> TensorViewer<'t, T> {
    pub(crate) fn new(t: &'t Tensor<T>, p: Projection) -> Self {
        Self {
            t,
            p,
            min: t.data().iter().fold(T::MAX, |ax, x| ax.cmp_min(x)),
            max: t.data().iter().fold(T::MIN, |ax, x| ax.cmp_max(x)),
        }
    }

    pub fn set_projection(&mut self, p: Projection) {
        self.p = p;
    }

    pub fn intro(&self, _term: Term) {}

    pub fn render(&self, term: Term) -> anyhow::Result<()> {
        let title = format!(
            "Shape: {}, projection: {}",
            self.t.shape(),
            (0..self.t.shape().rank())
                .map(|d| format!(
                    "{}: {}",
                    Projection::dim_to_char(d).unwrap(),
                    if d == self.p.x {
                        "X".to_string()
                    } else if d == self.p.y {
                        "Y".to_string()
                    } else {
                        self.p.fixed[d].to_string()
                    }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );

        BarcodeViewer::new(self.t, &self.p, self.min, self.max, &title).render(term)?;
        Histogram::new(self.t).send_to_gnuplot(term)?;

        Ok(())
    }
}

/// An histogram bucket. Histograms are represented as list of (consequent) buckets.
#[allow(dead_code)]
pub(crate) struct Bucket {
    /// The lower bound of the bucket range.
    pub min: f32,
    /// The higher bound of the bucket range.
    pub max: f32,
    /// The number of elements found in this bucket.
    pub count: usize,
}

/// An histogram generator for tensor values.
pub(crate) struct Histogram<'t, T> {
    /// A reference to the tensor to generate histograms for.
    t: &'t Tensor<T>,
    /// The cached value of the max. of the tensor values.
    min: T,
    /// The cached value of the min. of the tensor values.
    max: T,
}
impl<'t, T: Number> Histogram<'t, T> {
    pub fn new(t: &'t Tensor<T>) -> Self {
        let min = t.data().iter().fold(T::MAX, |ax, x| ax.cmp_min(x));
        let max = t.data().iter().fold(T::MIN, |ax, x| ax.cmp_max(x));
        Self { t, min, max }
    }

    // Generate an histogram for the wrapped tensor with the given number of buckets.
    // pub fn project(&self, bucket_count: usize) -> Vec<Bucket> {
    //     let step = self.span() / bucket_count as f32;

    //     (0..bucket_count)
    //         .map(|i| {
    //             let i = i as f32;
    //             let span = (
    //                 self.min.to_f32().unwrap() + i * step,
    //                 self.min.to_f32().unwrap() + (i + 1.) * step,
    //             );
    //             Bucket {
    //                 min: span.0,
    //                 max: span.1,
    //                 count: self
    //                     .t
    //                     .data()
    //                     .iter()
    //                     .filter(|x| x.to_f32().unwrap() >= span.0 && x.to_f32().unwrap() < span.1)
    //                     .count(),
    //             }
    //         })
    //         .collect::<Vec<_>>()
    // }

    // The span of values contained by the tensor.
    // fn span(&self) -> f32 {
    //     self.max.to_f32().unwrap() - self.min.to_f32().unwrap()
    // }

    // pub fn small(&self) -> String {
    //     const MIN_AVG_PER_BUCKET: usize = 5;
    //     let (w, _) = term_size::dimensions().unwrap();
    //     let buckets = self.project(w.min(self.t.data().len() / MIN_AVG_PER_BUCKET));
    //     let b_max = buckets.iter().map(|b| b.count).max().unwrap();
    //     let mut r = String::with_capacity(buckets.len());
    //     for b in buckets.iter() {
    //         let bb = 8 * b.count;
    //         let char = if bb == b_max {
    //             '█'
    //         } else if bb >= 7 * b_max {
    //             '▇'
    //         } else if bb >= 6 * b_max {
    //             '▆'
    //         } else if bb >= 5 * b_max {
    //             '▅'
    //         } else if bb >= 4 * b_max {
    //             '▄'
    //         } else if bb >= 3 * b_max {
    //             '▃'
    //         } else if bb >= 2 * b_max {
    //             '▂'
    //         } else if bb > 0 {
    //             '▁'
    //         } else {
    //             ' '
    //         };
    //         r.push(char);
    //     }
    //     r
    // }

    /// Render this histogram to the specified [`Term`].
    pub fn send_to_gnuplot(&self, term: Term) -> anyhow::Result<()> {
        with_gnuplot(|gp| {
            let title = format!(
                "shape: {} ({} elts.), range: [{:.2}, {:.2}]",
                self.t.shape(),
                self.t.data().len(),
                self.min,
                self.max,
            );
            let bin_count = 20;

            let gp_outfile = match term {
                Term::Tty => {
                    writeln!(gp, "set term dumb")?;
                    None
                }
                Term::Qt => {
                    writeln!(gp, "set term qt")?;
                    None
                }
                Term::Sixel => {
                    writeln!(gp, "set term sixelgd")?;
                    None
                }
                Term::Png => {
                    let out_file = tempfile::Builder::new()
                        .suffix(".png")
                        .disable_cleanup(true)
                        .tempfile()
                        .context("failed to create a temporary file")?;
                    writeln!(gp, "set term png size 1000, 1000")?;
                    writeln!(gp, "set output \"{}\"", out_file.path().display())?;
                    Some(out_file)
                }
            };
            writeln!(
                gp,
                r#"
set title "{title}"

n = {bin_count}
max = {}
min = {}
width = (max-min)/n

hist(x, width)=width * floor(x / width) + width / 2.0

set style fill solid 0.5
set log y
set boxwidth width

set xrange [min:max]
set yrange [0:]
set xlabel "Magnitude"
set ylabel "Count"
set tics out nomirror

set style fill solid 0.5

plot '-' u (hist($1, width)):(1.0) smooth freq with boxes lc 'skyblue' notitle
"#,
                self.max, self.min
            )?;

            // The raw data, one value par line.
            for x in self.t.data() {
                writeln!(gp, "{x:?}").unwrap();
            }

            Ok(gp_outfile)
        })
    }
}
