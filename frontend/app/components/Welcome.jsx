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
            This is a <strong>demonstration project</strong> to predict cognitive decline risk
            from speech patterns. It is for <strong>demo and research purposes only</strong> and
            <strong> should not replace</strong> professional medical advice or NHS memory clinic assessments.
          </p>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>This tool is <strong>not a clinical diagnosis</strong>.</li>
            <li>All data is processed in real-time and <strong>not stored</strong>.</li>
            <li>To minimise costs, <strong>only 3 attempts are allowed per IP address</strong>.</li>
            <li>Your privacy is respected; no personal data is saved.</li>
          </ul>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            By using this service, you agree to these terms.
          </p>
        </div>

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