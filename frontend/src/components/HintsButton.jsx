import React, { useState } from "react";
import "./HintsButton.css";

function LightbulbIcon() {
  return (
    <svg
      className="game-hint-icon"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      aria-hidden="true"
    >
      <path
        d="M9 18h6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M10 22h4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M8.6 14.4C7.3 13.3 6.5 11.7 6.5 10a5.5 5.5 0 0 1 11 0c0 1.7-.8 3.3-2.1 4.4-.8.7-1.4 1.4-1.4 2.6h-4c0-1.2-.6-1.9-1.4-2.6Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function HintsButton({ className = "" }) {
  const [showHints, setShowHints] = useState(false);

  return (
    <>
      <button
        className={`game-hint-button ${className}`}
        onClick={() => setShowHints(true)}
        aria-label="Show game hints"
      >
        <LightbulbIcon />
        <span>Hints</span>
      </button>

      {showHints && (
        <div className="game-hint-overlay" onClick={() => setShowHints(false)}>
          <div className="game-hint-modal" onClick={(e) => e.stopPropagation()}>
            <button
              className="game-hint-close"
              onClick={() => setShowHints(false)}
              aria-label="Close hints"
            >
              ×
            </button>

            <div className="game-hint-modal-icon">
              <LightbulbIcon />
            </div>

            <h2>Player Hints</h2>

            <p className="game-hint-intro">
              Strong station locations usually have a mix of people, activity,
              and nearby destinations.
            </p>

            <ul className="game-hint-list">
              <li>Look for highly populated areas.</li>
              <li>Areas with younger residents (college students!) may perform better.</li>
              <li>Lower-income areas may indicate stronger transportation need.</li>
              <li>Try locations near transit stops, jobs, parks, restaurants, or UT.</li>
              <li>Avoid placing stations too far away from the existing network.</li>
            </ul>
          </div>
        </div>
      )}
    </>
  );
}