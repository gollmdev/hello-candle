use clap::Parser;

/// 一个简单的 CLI 示例
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// 用户名
    #[arg(short, long)]
    name: String,

    /// 年龄
    #[arg(short, long)]
    age: u32,
}

fn main() {
    let args = Args::parse();

    println!("name = {}", args.name);
    println!("age = {}", args.age);
}

// cargo run --bin hello --  --name aaa --age 56