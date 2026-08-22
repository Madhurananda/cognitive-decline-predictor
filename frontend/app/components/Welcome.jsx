'use client';

export default function Welcome({ onAccept }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 p-4 transition-colors duration-300">
      {/* Card */}
      <div className="max-w-2xl w-full card shadow-xl border-0">
        <h1 className="text-3xl sm:text-4xl font-bold text-center text-gray-800 dark:text-white mb-2">
          🧠 Cognitive Decline Predictor
        </h1>
        <p className="text-center text-sm text-gray-500 dark:text-gray-400 mb-6">
          A research demonstration tool
        </p>

        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          
          <p>
            This project explores whether patterns in everyday speech can help identify
            characteristics associated with early cognitive decline. It was developed
            using data from <strong>PROCESS-2</strong>, a benchmark speech corpus derived
            from the <strong>CognoSpeak</strong> research programme, which investigates
            the automatic and remote assessment of cognitive decline from real-world
            conversational speech.
          </p>

          <p>
            The aim is to demonstrate how research data, speech and language processing,
            machine learning, and modern AI technologies can be combined into an
            end-to-end interactive system.
          </p>

          <p>
            This is a <strong>demonstration project</strong> for research and educational
            purposes only. It <strong>is not a clinical diagnostic tool</strong> and
            <strong> should not replace</strong> professional medical advice or NHS
            memory clinic assessments.
          </p>

          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>This tool is <strong>not a clinical diagnosis</strong>.</li>
            <li>All data is processed in real-time and <strong>not stored</strong>.</li>
            <li>To minimise costs, <strong>only 5 attempts are allowed per IP address per day</strong>.</li>
            <li>Your privacy is respected; no personal data is saved.</li>
          </ul>

          {/* Warm-up notice */}
          <div className="mt-2 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg text-sm text-amber-700 dark:text-amber-300">
            <span className="font-medium">⏳ Warm‑up time:</span> The first request after a period of inactivity may take up to <strong>20 seconds</strong> while the cloud service starts up (free tier with 0 minimum instances). Please be patient – subsequent requests will be faster.
          </div>

          <p className="text-sm text-gray-500 dark:text-gray-400">
            By using this service, you agree to these terms.
          </p>
        </div>

        {/* ---- Research Background (Collapsible) ---- */}
        <details className="mt-6 border-t border-gray-200 dark:border-gray-700 pt-4 group">
          <summary className="text-sm font-medium text-gray-600 dark:text-gray-300 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors list-none flex items-center gap-2">
            <span className="text-base">📄</span> Research background
            <span className="ml-auto text-xs text-gray-400 group-open:rotate-180 transition-transform">▼</span>
          </summary>
          <div className="mt-3 text-sm text-gray-500 dark:text-gray-400 space-y-1">
            <ul className="list-none space-y-1">
              <li>
                <a
                  href="https://ieeexplore.ieee.org/abstract/document/10969487"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech (2025)
                </a>
              </li>
              <li>
                <a
                  href="https://ieeexplore.ieee.org/abstract/document/11284700"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  Automatic detection of early cognitive decline using multimodal feature fusion and transfer learning on real-world conversational speech (2025)
                </a>
              </li>
              <li>
                <a
                  href="https://arxiv.org/abs/2605.14888"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                >
                  PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection (2026)
                </a>
              </li>
            </ul>
            <p className="mt-2 text-xs text-gray-400">
              These publications provide the scientific foundation for this demonstration.
            </p>
          </div>
        </details>

        <button
          onClick={onAccept}
          className="w-full mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          I Agree & Continue
        </button>
      </div>

      {/* Footer - matches page.js */}
      <footer className="w-full max-w-4xl mt-8 text-center text-sm text-gray-500 dark:text-gray-400">
        Made with ❤️ by{' '}
        <a
          href="https://madhurananda.github.io/"
          target="_blank"
          rel="noopener noreferrer"
          className="footer-link"
        >
          Madhu Pahar
        </a>
        {' '}·{' '}
        <span className="text-xs">
          Demo only · Not for clinical diagnosis
        </span>
      </footer>
    </div>
  );
}