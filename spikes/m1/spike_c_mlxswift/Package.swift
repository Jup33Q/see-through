// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "spike_c_mlxswift",
  platforms: [.macOS(.v14)],
  dependencies: [
    // 本机 GitHub 子模块克隆不稳定，mlx-swift 使用本地 vendor 副本
    // (main @ 0bb916c, 2026-07-01, 含 Source/Cmlx/mlx 与 mlx-c 子模块内容)
    .package(path: "vendor/mlx-swift")
  ],
  targets: [
    .executableTarget(
      name: "unetbench",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
      ],
      path: "Sources/unetbench"
    )
  ]
)
