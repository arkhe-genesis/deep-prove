use std::ops::Range;

const DATA_SIZE_POWS: Range<i32> = 7..14;

#[derive(Debug, Copy, Clone)]
struct Args {
    pow2: i32,
}

fn default_sizes() -> impl Iterator<Item = Args> {
    DATA_SIZE_POWS.map(|pow2| Args { pow2 })
}

#[divan::bench_group]
mod add_op {
    use zkml::{Element, Shape, Tensor};

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let left = Tensor::<Element>::random(&shape);
        let right = Tensor::<Element>::random(&shape);
        bencher.bench(|| left.add(&right));
    }
}

#[divan::bench_group]
mod mul_op {
    use zkml::{Element, Shape, Tensor};

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let left = Tensor::<Element>::random(&shape);
        let right = Tensor::<Element>::random(&shape);
        bencher.bench(|| left.mul(&right));
    }
}

fn main() {
    divan::main();
}
