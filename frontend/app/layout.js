import './globals.css';

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>Cognitive Decline Predictor</title>
        <meta name="description" content="Predict cognitive decline from voice recordings" />
        <link rel="icon" href="/favicon.ico" />
        <meta name="grammarly" content="disabled" />
      </head>
      <body suppressHydrationWarning={true}>{children}</body>
    </html>
  );
}
