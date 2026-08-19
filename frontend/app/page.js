'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import Welcome from './components/Welcome';
import SpiderChart from './components/SpiderChart';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [accepted, setAccepted] = useState(false);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [inputSource, setInputSource] = useState(null);
  
  // New state for progress
  const [progress, setProgress] = useState(0);
  const [progressMessages, setProgressMessages] = useState([]);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const isRecordingRef = useRef(false);

  // Cleanup
  const cleanupRecording = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    isRecordingRef.current = false;
  }, []);

  useEffect(() => {
    return () => cleanupRecording();
  }, [cleanupRecording]);

  const cleanExplanation = (text) => {
    if (!text) return '';
    return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  };

  const getRiskDetails = (score) => {
    if (score < 0.5) {
      return { label: 'Low Risk', color: 'text-green-700 dark:text-green-300', bg: 'bg-green-100 dark:bg-green-900/30' };
    } else if (score < 0.65) {
      return { label: 'Low-Medium', color: 'text-yellow-700 dark:text-yellow-300', bg: 'bg-yellow-100 dark:bg-yellow-900/30' };
    } else if (score < 0.8) {
      return { label: 'Medium-High', color: 'text-orange-700 dark:text-orange-300', bg: 'bg-orange-100 dark:bg-orange-900/30' };
    } else {
      return { label: 'High', color: 'text-red-700 dark:text-red-300', bg: 'bg-red-100 dark:bg-red-900/30' };
    }
  };

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;

    if (!analyser || !dataArray) {
      animationRef.current = requestAnimationFrame(drawWaveform);
      return;
    }

    const draw = () => {
      if (!isRecordingRef.current) {
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#9ca3af';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Press "Start Answering" to begin', width / 2, height / 2 + 5);
        animationRef.current = requestAnimationFrame(draw);
        return;
      }

      analyser.getByteTimeDomainData(dataArray);
      
      const width = canvas.width;
      const height = canvas.height;
      
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, width, height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#3b82f6';
      ctx.beginPath();

      const sliceWidth = width / dataArray.length;
      let x = 0;

      for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * height / 2;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.stroke();

      ctx.beginPath();
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      ctx.setLineDash([]);

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([blob], 'recording.wav', { type: 'audio/wav' });
        setFile(audioFile);
        setInputSource('recorded');
        setRecordingDuration(0);
        cleanupRecording();
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      isRecordingRef.current = true;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      const dataArray = new Uint8Array(analyser.fftSize);
      dataArrayRef.current = dataArray;

      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = 800;
        canvas.height = 120;
        canvas.style.width = '100%';
        canvas.style.height = '120px';
      }

      drawWaveform();

      let seconds = 0;
      timerRef.current = setInterval(() => {
        seconds++;
        setRecordingDuration(seconds);
        if (seconds >= 180) stopRecording();
      }, 1000);

    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.');
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecordingRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please record audio or select a file.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setProgressMessages([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use the streaming endpoint
      const response = await fetch(`${API_URL}/predict-stream`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(6);
            try {
              const data = JSON.parse(jsonStr);
              
              if (data.step === 'error') {
                setError(data.message);
                setLoading(false);
                return;
              }
              
              if (data.step === 'complete') {
                setResult(data.result);
                setLoading(false);
                return;
              }
              
              // Update progress
              setProgress(data.progress || 0);
              
              // Add message if not already present
              setProgressMessages(prev => {
                const exists = prev.some(msg => msg.step === data.step);
                if (exists) return prev;
                return [...prev, { step: data.step, message: data.message, progress: data.progress }];
              });
              
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (!accepted) {
    return <Welcome onAccept={() => setAccepted(true)} />;
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-start p-4 sm:p-6 md:p-8 bg-gray-50 dark:bg-gray-900">
      <div className="w-full max-w-4xl card">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center mb-2">
          🧠 Cognitive Decline Predictor
        </h1>
        <p className="text-center text-gray-600 dark:text-gray-400 mb-6 text-sm sm:text-base">
          Please describe the following Cookie Theft picture (between 10 seconds and 3 minutes)
        </p>

        <div className="flex justify-center mb-6">
          <img
            src="/images/cookie_theft.png"
            alt="Cookie Theft picture – describe what you see"
            className="w-full max-w-md rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
            onError={(e) => {
              e.target.src = 'https://via.placeholder.com/400x300?text=Cookie+Theft+Image';
              e.target.onerror = null;
            }}
          />
        </div>

        {/* Centered Start Answering button */}
        <div className="flex flex-col items-center gap-4 mb-4">
          <div className="flex items-center gap-4">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-full sm:w-auto px-6 py-2 rounded-lg text-white font-semibold transition ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
              aria-label={isRecording ? 'Stop recording' : 'Start answering'}
            >
              {isRecording ? `⏹ Stop (${recordingDuration}s)` : '🎙 Start Answering'}
            </button>
            {file && (
              <span className="text-sm text-gray-600 dark:text-gray-400">
                ✅ Ready ({Math.round(file.size / 1024)} KB)
              </span>
            )}
          </div>
        </div>

        <div className="mb-4">
          <canvas
            ref={canvasRef}
            className="w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800"
            style={{ height: '120px', display: 'block' }}
            aria-label="Live audio waveform"
          />
          {isRecording && (
            <p className="text-xs text-green-600 dark:text-green-400 text-center mt-1">🔴 Live waveform</p>
          )}
          {!isRecording && !file && (
            <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-1">Press "Start Answering" to begin</p>
          )}
          {!isRecording && file && (
            <p className="text-xs text-blue-600 dark:text-blue-400 text-center mt-1">✅ Recording ready for analysis</p>
          )}
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Or upload a file:
          </label>
          <input
            type="file"
            accept=".wav,.mp3,.m4a"
            onChange={(e) => {
              const selectedFile = e.target.files[0];
              if (selectedFile) {
                setFile(selectedFile);
                setInputSource('uploaded');
              }
            }}
            className="block w-full text-sm text-gray-500 dark:text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 dark:file:bg-blue-900/30 file:text-blue-700 dark:file:text-blue-300 hover:file:bg-blue-100 dark:hover:file:bg-blue-800/40 transition"
            aria-label="Choose an audio file"
          />
        </div>

        <form onSubmit={handleSubmit}>
          <button
            type="submit"
            disabled={loading || !file}
            className="w-full py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Processing...
              </span>
            ) : (
              '🔮 Get Results'
            )}
          </button>
        </form>

        {/* Progress display */}
        {loading && (
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-3 mb-3">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent" />
              <span className="text-blue-600 dark:text-blue-400 font-medium">Processing...</span>
              <span className="text-sm text-blue-500 dark:text-blue-300 ml-auto">{progress}%</span>
            </div>
            
            {/* Progress bar */}
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mb-3">
              <div 
                className="h-full bg-blue-600 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            
            {/* Step messages */}
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {progressMessages.map((msg, idx) => (
                <div key={idx} className="text-sm text-gray-600 dark:text-gray-300 flex items-center gap-2">
                  <span className="text-xs text-blue-500">▸</span>
                  {msg.message}
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-red-600 dark:text-red-400 font-medium">❌ Something went wrong</p>
            <p className="text-red-500 dark:text-red-300 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-sm text-red-600 dark:text-red-400 hover:underline"
              aria-label="Dismiss error"
            >
              Dismiss
            </button>
          </div>
        )}

        {result && (
          <div className="mt-6 space-y-6 animate-fadeIn">
            <h2 className="text-xl sm:text-2xl font-bold">📊 Results</h2>

            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-6 space-y-4">
              <div className="text-center">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Risk Assessment</p>
                {(() => {
                  const risk = getRiskDetails(result.risk_score);
                  return (
                    <div className={`mt-1 inline-block px-6 py-3 rounded-xl text-2xl sm:text-3xl font-extrabold ${risk.bg} ${risk.color}`}>
                      {risk.label} ({ (result.risk_score * 100).toFixed(1) }%)
                    </div>
                  );
                })()}
              </div>

              <div>
                <span className="font-medium">Transcript:</span>
                <p className="mt-1 text-gray-600 dark:text-gray-300 italic leading-relaxed">
                  "{result.transcript}"
                </p>
              </div>

              <div>
                <span className="font-medium">Explanation:</span>
                <p className="mt-1 text-gray-600 dark:text-gray-300 leading-relaxed">
                  {cleanExplanation(result.explanation)}
                </p>
              </div>
            </div>

            {result.spider_data && (
              <div>
                <h3 className="text-2xl font-bold">📈 Analysis</h3>
                <div className="mt-2 flex justify-center">
                  <SpiderChart spiderData={result.spider_data} />
                </div>
                <div className="mt-4 text-sm text-gray-500 dark:text-gray-400 text-center">
                  <p className="font-medium">What do the corners mean?</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2 text-xs">
                    <div><span className="font-semibold">Pause Ratio</span> ↑ = more pauses (worse)</div>
                    <div><span className="font-semibold">Vocabulary Diversity</span> ↓ = fewer unique words (worse)</div>
                    <div><span className="font-semibold">Content Words</span> ↓ = fewer meaningful words (worse)</div>
                    <div><span className="font-semibold">Sentence Length</span> ↓ = shorter sentences (worse)</div>
                    <div><span className="font-semibold">Word Count</span> ↓ = less verbal output (worse)</div>
                    <div><span className="font-semibold">Pitch Variation</span> ↓ = less expressiveness (worse)</div>
                  </div>
                  <p className="mt-2 text-xs text-gray-400">Higher scores are better (closer to healthy average).</p>
                </div>
              </div>
            )}

            <details className="mt-4">
              <summary className="cursor-pointer text-blue-600 dark:text-blue-400 hover:underline font-medium">
                📋 Show detailed predictions
              </summary>
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 overflow-auto rounded-lg mt-2 text-sm">
                {JSON.stringify(result.features, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>

      <div className="w-full max-w-4xl mt-6 flex flex-col sm:flex-row items-center justify-center gap-4 text-sm text-gray-600 dark:text-gray-400">
        <span className="font-medium">💬 Feedback:</span>
        <a
          href="https://github.com/madhurananda/cognitive-decline-predictor"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 px-3 py-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition"
          aria-label="Star this repository on GitHub"
        >
          ⭐ Star this repo
        </a>
        <a
          href="https://github.com/madhurananda/cognitive-decline-predictor/issues/new"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 px-3 py-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition"
          aria-label="Create an issue on GitHub"
        >
          🐛 Create an issue
        </a>
        <span className="text-xs text-gray-400 dark:text-gray-500">
          Your feedback helps improve this demo
        </span>
      </div>

      <footer className="w-full max-w-4xl mt-4 text-center text-sm text-gray-500 dark:text-gray-400">
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

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}