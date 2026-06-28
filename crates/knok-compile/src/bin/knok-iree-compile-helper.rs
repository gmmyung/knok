use std::{
    env, fs,
    io::{self, Read},
    path::PathBuf,
    process,
    sync::{Mutex, OnceLock},
};

use eerie::compiler::{Compiler, MemBufferOutput, Pipeline};

static COMPILER: OnceLock<Result<Mutex<Compiler>, String>> = OnceLock::new();

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("revision") => {
            println!("{}", compiler_revision()?);
            Ok(())
        }
        Some("compile") => {
            let mut output = None;
            let mut flags = Vec::new();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--output" => output = args.next().map(PathBuf::from),
                    "--" => {
                        flags.extend(args);
                        break;
                    }
                    _ => anyhow::bail!("unknown argument `{arg}`"),
                }
            }
            let output = output.ok_or_else(|| anyhow::anyhow!("missing --output path"))?;
            let mut mlir = Vec::new();
            io::stdin().read_to_end(&mut mlir)?;
            let vmfb = compile_with_eerie(&flags, &mlir)?;
            fs::write(output, vmfb)?;
            Ok(())
        }
        Some(command) => anyhow::bail!("unknown command `{command}`"),
        None => anyhow::bail!("expected `revision` or `compile` command"),
    }
}

fn global_compiler() -> anyhow::Result<&'static Mutex<Compiler>> {
    COMPILER
        .get_or_init(|| {
            Compiler::new()
                .map(Mutex::new)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

fn compiler_revision() -> anyhow::Result<String> {
    let compiler = global_compiler()?
        .lock()
        .map_err(|_| anyhow::anyhow!("IREE compiler lock poisoned"))?;
    Ok(compiler
        .get_revision()
        .map_err(|error| anyhow::anyhow!("{error}"))?)
}

fn compile_with_eerie(flags: &[String], mlir: &[u8]) -> anyhow::Result<Vec<u8>> {
    let compiler = global_compiler()?
        .lock()
        .map_err(|_| anyhow::anyhow!("IREE compiler lock poisoned"))?;
    let mut session = compiler.create_session();
    session
        .set_flags(flags.to_vec())
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let source = session
        .create_source_from_buf(mlir)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut invocation = session.create_invocation();
    let mut output = MemBufferOutput::new(&compiler).map_err(|error| anyhow::anyhow!("{error}"))?;
    invocation
        .parse_source(source)
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .set_verify_ir(true)
        .set_compile_to_phase("end")
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .pipeline(Pipeline::Std)
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .output_vm_byte_code(&mut output)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(output
        .map_memory()
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .to_vec())
}
