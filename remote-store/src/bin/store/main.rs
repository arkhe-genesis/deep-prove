use anyhow::bail;
use clap::Parser;
use remote_store::server;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version = remote_store::get_version!(), about)]
struct Args {
    #[arg(long, env, default_value = "4001")]
    port: u16,
    #[arg(long, env)]
    store_dir: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Args { port, store_dir } = Args::parse();

    if !store_dir.exists() {
        bail!(
            "The given store-dir \"{}\" doesn't exist",
            store_dir.display()
        );
    }

    server::run(store_dir, port).await
}
