'use client';

import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const SpiderChart = ({ spiderData }) => {
  if (!spiderData) return <p className="text-center text-gray-500">No data available</p>;

  const featureLabels = {
    silence_ratio: 'Pause Ratio',
    ttr: 'Vocabulary Diversity',
    content_ratio: 'Content Words',
    mlu: 'Sentence Length',
    word_count: 'Word Count',
    pitch_std: 'Pitch Variation',
  };

  const labels = Object.keys(spiderData).map((key) => featureLabels[key] || key);
  const userScores = Object.values(spiderData).map((d) => d.normalized_score);
  const healthyBaseline = labels.map(() => 0.7);

  const data = {
    labels,
    datasets: [
      {
        label: 'You',
        data: userScores,
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 4,           // ← Bolder lines
        pointRadius: 6,           // ← Larger dots
        pointBorderWidth: 2,
        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
        pointBorderColor: '#fff',
        pointHoverRadius: 8,
      },
      {
        label: 'Healthy Average',
        data: healthyBaseline,
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        borderColor: 'rgba(75, 192, 192, 0.8)',
        borderDash: [5, 5],
        borderWidth: 3,           // ← Slightly bolder dashed line
        pointRadius: 4,
        pointBorderWidth: 2,
        pointBackgroundColor: 'rgba(75, 192, 192, 0.8)',
        pointBorderColor: '#fff',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      r: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.2,
          backdropColor: 'transparent',
          font: { size: 10 },
        },
        pointLabels: {
          font: { size: 13, weight: 'bold' }, // ← Bolder labels
        },
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { size: 14, weight: 'bold' },
          usePointStyle: true,
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const label = context.dataset.label || '';
            const value = context.parsed.r;
            const feature = context.label;
            const data = spiderData[Object.keys(spiderData)[context.dataIndex]];
            if (data) {
              return [
                `${label}: ${(value * 100).toFixed(0)}%`,
                `Your value: ${data.user_value.toFixed(3)}`,
                `Healthy mean: ${data.healthy_mean.toFixed(3)}`,
                `Z-score: ${data.z_score.toFixed(2)}`,
              ];
            }
            return `${label}: ${(value * 100).toFixed(0)}%`;
          },
        },
      },
    },
  };

  return (
    <div className="w-full max-w-2xl mx-auto aspect-square">
      <Radar data={data} options={options} />
    </div>
  );
};

export default SpiderChart;