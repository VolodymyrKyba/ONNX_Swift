import Foundation
import onnxruntime_objc

class Tokenizer {
    private var vocab: [String: Int]

    init(vocab: [String: Int]) {
        self.vocab = vocab
    }

    func tokenize(text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map { String($0) }
        return words.map { vocab[$0] ?? vocab["<OOV>"] ?? 1 }
    }
}

class LabelVocabLoader {
    private(set) var labelMap: [Int: String] = [:]
    private(set) var vocab: [String: Int] = [:]

    init(labelMapPath: String, vocabPath: String) throws {
        let labelMapData = try Data(contentsOf: URL(fileURLWithPath: labelMapPath))
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))

        if let labelMapJson = try JSONSerialization.jsonObject(with: labelMapData) as? [String: String] {
            for (key, value) in labelMapJson {
                if let intKey = Int(key) {
                    labelMap[intKey] = value
                }
            }
        }

        if let vocabJson = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
            vocab = vocabJson
        }
    }
}

class ONNXModelRunner {
    private var session: ORTSession
    private var labelMap: [Int: String]
    private var vocab: [String: Int]
    private let maxLen = 30

    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
              let labelPath = Bundle.main.path(forResource: "label_map", ofType: "json"),
              let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json") else {
            throw NSError(domain: "Paths", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing model or json files"])
        }

        let loader = try LabelVocabLoader(labelMapPath: labelPath, vocabPath: vocabPath)
        self.labelMap = loader.labelMap
        self.vocab = loader.vocab

        let env = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)

        print("✅ Model loaded.")
        print("Input names: \(try session.inputNames())")
        print("Output names: \(try session.outputNames())")
    }

    func predict(text: String) throws -> [(String, Float)] {
        let tokenizer = Tokenizer(vocab: vocab)
        var tokens = tokenizer.tokenize(text: text)

        if tokens.count < maxLen {
            tokens += Array(repeating: 0, count: maxLen - tokens.count)
        } else if tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }

        let int32Tokens = tokens.map { Int32($0) }
        let inputData = Data(from: int32Tokens)

        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .int32,
            shape: [1, NSNumber(value: maxLen)]
        )

        let inputs: [String: ORTValue] = [
            "input": inputTensor
        ]

        let outputs = try session.run(
            withInputs: inputs,
            outputNames: Set(["sequential"]),
            runOptions: nil
        )

        guard let ortValue = outputs["sequential"] else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'sequential' output"])
        }

        let rawData = try ortValue.tensorData()

        let resultData: Data
        if let d = rawData as? Data {
            resultData = d
        } else {
            throw NSError(domain: "ONNX", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output is not valid Data"])
        }

        let classCount = labelMap.count
        let scores: [Float] = resultData.withUnsafeBytes { buffer in
            let pointer = buffer.bindMemory(to: Float.self)
            return Array(pointer.prefix(classCount))
        }

        return scores.enumerated().map { (i, score) in
            (labelMap[i] ?? "Unknown", score)
        }
    }
    
    // New function for detailed prediction with performance metrics
    func predictDetailed(text: String) throws -> PredictionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let preprocessStart = startTime
        
        let tokenizer = Tokenizer(vocab: vocab)
        var tokens = tokenizer.tokenize(text: text)

        if tokens.count < maxLen {
            tokens += Array(repeating: 0, count: maxLen - tokens.count)
        } else if tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }

        let int32Tokens = tokens.map { Int32($0) }
        let inputData = Data(from: int32Tokens)
        let preprocessTime = (CFAbsoluteTimeGetCurrent() - preprocessStart) * 1000

        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .int32,
            shape: [1, NSNumber(value: maxLen)]
        )

        let inputs: [String: ORTValue] = [
            "input": inputTensor
        ]

        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let outputs = try session.run(
            withInputs: inputs,
            outputNames: Set(["sequential"]),
            runOptions: nil
        )
        let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000

        guard let ortValue = outputs["sequential"] else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'sequential' output"])
        }

        let postprocessStart = CFAbsoluteTimeGetCurrent()
        let rawData = try ortValue.tensorData()

        let resultData: Data
        if let d = rawData as? Data {
            resultData = d
        } else {
            throw NSError(domain: "ONNX", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output is not valid Data"])
        }

        let classCount = labelMap.count
        let scores: [Float] = resultData.withUnsafeBytes { buffer in
            let pointer = buffer.bindMemory(to: Float.self)
            return Array(pointer.prefix(classCount))
        }

        let predictions = scores.enumerated().map { (i, score) in
            (labelMap[i] ?? "Unknown", score)
        }
        
        let postprocessTime = (CFAbsoluteTimeGetCurrent() - postprocessStart) * 1000
        let totalTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        // Find best prediction
        let bestPrediction = predictions.max { $0.1 < $1.1 }
        
        let result = PredictionResult(
            inputText: text,
            predictions: predictions.sorted { $0.1 > $1.1 },
            bestPrediction: bestPrediction,
            totalTime: totalTime,
            preprocessTime: preprocessTime,
            inferenceTime: inferenceTime,
            postprocessTime: postprocessTime
        )
        
        // Print detailed results to console/terminal
        print("\n" + result.formattedSummary + "\n")
        
        return result
    }
}

// Structure to hold detailed prediction results
struct PredictionResult {
    let inputText: String
    let predictions: [(String, Float)]
    let bestPrediction: (String, Float)?
    let totalTime: Double
    let preprocessTime: Double
    let inferenceTime: Double
    let postprocessTime: Double
    
    var formattedSummary: String {
        let best = bestPrediction ?? ("Unknown", 0.0)
        let confidence = best.1 * 100
        
        let probabilities = predictions.map { label, score in
            let percentage = score * 100
            let bar = String(repeating: "█", count: Int(percentage / 5)) // Scale bars
            let star = score == best.1 ? " ⭐" : ""
            return String(format: "   📝 %@: %.1f%% %@%@", label, percentage, bar, star)
        }.joined(separator: "\n")
        
        return """
        🤖 ONNX TEXT CLASSIFIER - iOS IMPLEMENTATION
        =============================================
        🔄 Processing: \(inputText)
        
        💻 SYSTEM INFORMATION:
           Platform: iOS
           Implementation: Swift with ONNX Runtime
        
        📊 CLASSIFICATION RESULTS:
        ⏱️  Processing Time: \(String(format: "%.1f", totalTime))ms
           🏆 Predicted Category: \(best.0.uppercased()) 📝
           📈 Confidence: \(String(format: "%.1f", confidence))%
           📝 Input Text: "\(inputText)"
        
        📊 DETAILED PROBABILITIES:
        \(probabilities)
        
        📈 PERFORMANCE SUMMARY:
           Total Processing Time: \(String(format: "%.2f", totalTime))ms
           ┣━ Preprocessing: \(String(format: "%.2f", preprocessTime))ms (\(String(format: "%.1f", (preprocessTime/totalTime)*100))%)
           ┣━ Model Inference: \(String(format: "%.2f", inferenceTime))ms (\(String(format: "%.1f", (inferenceTime/totalTime)*100))%)
           ┗━ Post-processing: \(String(format: "%.2f", postprocessTime))ms (\(String(format: "%.1f", (postprocessTime/totalTime)*100))%)
           🚀 Throughput: \(String(format: "%.1f", 1000/totalTime)) texts/sec
           Performance Rating: \(totalTime < 50 ? "✅ EXCELLENT" : totalTime < 100 ? "✅ GOOD" : "⚠️ FAIR")
        """
    }
}

// 🔁 Розширення Data
private extension Data {
    init<T>(from array: [T]) {
        self = array.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
