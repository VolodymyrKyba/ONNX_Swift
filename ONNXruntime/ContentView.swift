import SwiftUI

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var isLoading: Bool = false
    @State private var lastProcessed: String = ""
    @State private var predictions: [(String, Float)] = []
    @State private var detailedResult: String = ""
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 20) {
            Text("🤖 ONNX Text Classifier")
                .font(.title)
                .padding(.top)

            VStack(spacing: 15) {
                TextField("Enter your text here", text: $inputText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)

                Button(action: classifyText) {
                    Text(isLoading ? "🔄 Processing..." : "🚀 Classify Text")
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(isLoading ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(isLoading || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .padding(.horizontal)
            }
            
            if let error = errorMessage {
                Text("❌ \(error)")
                    .foregroundColor(.red)
                    .padding()
            }

            if !detailedResult.isEmpty {
                ScrollView {
                    Text(detailedResult)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                }
                .frame(maxHeight: 300)
            }

            if !predictions.isEmpty {
                Text("📊 Prediction Results:")
                    .font(.headline)
                    .padding(.top, 10)
                
                List(predictions, id: \.0) { item in
                    HStack {
                        Text(item.0)
                            .fontWeight(.bold)
                        Spacer()
                        Text(String(format: "%.1f%%", item.1 * 100))
                            .foregroundColor(item.1 == predictions.first?.1 ? .green : .primary)
                    }
                }
                .frame(maxHeight: 150)
            }

            Spacer()
        }
        .padding()
    }

    private func classifyText() {
        isLoading = true
        errorMessage = nil
        detailedResult = ""
        predictions = []
        let textToProcess = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Console output separator
        print("\n" + String(repeating: "=", count: 60))
        print("🚀 NEW CLASSIFICATION REQUEST")
        print(String(repeating: "=", count: 60))
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let runner = try ONNXModelRunner()
                let result = try runner.predictDetailed(text: textToProcess)
                
                DispatchQueue.main.async {
                    // Update app UI with results
                    self.lastProcessed = textToProcess
                    self.predictions = result.predictions
                    self.detailedResult = result.formattedSummary
                    self.errorMessage = nil
                    self.isLoading = false
                    
                    // Console completion marker
                    print(String(repeating: "=", count: 60))
                    print("✅ CLASSIFICATION COMPLETE - Results displayed in app")
                    print(String(repeating: "=", count: 60) + "\n")
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.predictions = []
                    self.detailedResult = ""
                    self.isLoading = false
                    print("❌ ERROR: \(error.localizedDescription)")
                    print(String(repeating: "=", count: 60) + "\n")
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
