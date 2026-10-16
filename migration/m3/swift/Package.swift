// swift-tools-version: 6.0
import PackageDescription

// ldrunner: M3–M5 CoreML 迁移与原生后处理的命令行对齐工具。
// 零外部依赖：仅用 Foundation / CoreML / CoreGraphics / ImageIO / Accelerate。
let package = Package(
    name: "ldrunner",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "ldrunner",
            path: "Sources/ldrunner"
        )
    ]
)
