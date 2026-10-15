import Foundation

/// ldrunner — LayerDiff3D CoreML 迁移 M3 对齐工具
///
/// 用法:
///   ldrunner sched-check-fixture [--out x.json]
///   ldrunner sched-check  --traj <dir> [--out x.json]
///   ldrunner unet-step    --traj <dir> --step N [--out x.json]
///   ldrunner denoise      --traj <dir> [--out x.json]
///   ldrunner decode       --traj <dir> [--out x.json]
///   ldrunner e2e          --pass body|head [--out x.json]
///
/// 退出码: 0 正常/通过, 1 对比未通过, 2 用法错误或缺少前置产物(待锚点/待模型)

func usage() -> Never {
    print("""
    ldrunner — LayerDiff3D M3 CoreML 对齐工具
      sched-check-fixture                 调度器对 Fixtures 自检（无需锚点/模型）
      sched-check  --traj <dir>           逐步对齐 golden trajectory
      unet-step    --traj <dir> --step N  merged UNet 单步对比
      denoise      --traj <dir>           30 步 CoreML 去噪循环（对齐模式）
      decode       --traj <dir>           decode_in → PNG → golden 对比
      e2e          --pass body|head       全胶水（编码→去噪→解码→golden）
    公共参数: --out <path>（相对路径落在 migration/m3/out/ 下）
    """)
    exit(2)
}

let argv = Array(CommandLine.arguments.dropFirst())
guard let cmd = argv.first, !cmd.hasPrefix("--") else { usage() }

// 解析 --key value 参数
var params: [String: String] = [:]
var i = 1
while i < argv.count {
    if argv[i].hasPrefix("--"), i + 1 < argv.count {
        params[String(argv[i].dropFirst(2))] = argv[i + 1]
        i += 2
    } else { i += 1 }
}

let out = params["out"]

do {
    switch cmd {
    case "sched-check-fixture":
        try Commands.schedCheckFixture(out: out)
    case "sched-check":
        guard let traj = params["traj"] else { usage() }
        try Commands.schedCheck(traj: traj, out: out)
    case "unet-step":
        guard let traj = params["traj"], let n = params["step"].flatMap(Int.init) else { usage() }
        try Commands.unetStep(traj: traj, stepN: n, out: out)
    case "denoise":
        guard let traj = params["traj"] else { usage() }
        try Commands.denoise(traj: traj, out: out)
    case "decode":
        guard let traj = params["traj"] else { usage() }
        try Commands.decode(traj: traj, out: out)
    case "e2e":
        guard let p = params["pass"], ["body", "head"].contains(p) else { usage() }
        try Commands.e2e(pass: p, out: out)
    default:
        usage()
    }
} catch let e as Err {
    fputs("error: \(e.description)\n", stderr)
    exit(2)
} catch {
    fputs("error: \(error.localizedDescription)\n", stderr)
    exit(2)
}
