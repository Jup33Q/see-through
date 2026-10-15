// swift-tools-version: 6.0
import PackageDescription

// ldrunner: LayerDiff3D CoreML 迁移 M3 阶段的命令行对齐工具。
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
