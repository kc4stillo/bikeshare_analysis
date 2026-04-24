import React, { useEffect, useRef, useState } from "react";
import HintsButton from "./HintsButton";
import "./StartScreen.css";

export default function StartScreen({ onPlayStart, onTransitionComplete }) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [bubbleOrigin, setBubbleOrigin] = useState({
    x: "50vw",
    y: "50vh",
  });

  const playButtonRef = useRef(null);
  const transitionTimeoutRef = useRef(null);

  useEffect(() => {
    return () => {
      if (transitionTimeoutRef.current) {
        window.clearTimeout(transitionTimeoutRef.current);
      }
    };
  }, []);

  function handlePlayClick() {
    if (isTransitioning) return;

    const buttonRect = playButtonRef.current?.getBoundingClientRect();

    if (buttonRect) {
      setBubbleOrigin({
        x: `${buttonRect.left + buttonRect.width / 2}px`,
        y: `${buttonRect.top + buttonRect.height / 2}px`,
      });
    }

    setIsTransitioning(true);
    onPlayStart?.();

    transitionTimeoutRef.current = window.setTimeout(() => {
      onTransitionComplete?.();
    }, 1250);
  }

  return (
    <div
      className={`start-screen ${isTransitioning ? "is-transitioning" : ""}`}
      style={{
        "--bubble-x": bubbleOrigin.x,
        "--bubble-y": bubbleOrigin.y,
      }}
    >
      <header className="start-header">
        <a
          href="https://www.capmetro.org/bikeshare"
          target="_blank"
          rel="noreferrer"
          aria-label="Visit CapMetro Bikeshare"
          className="capmetro-logo-link"
        >
          <img
            src={`${process.env.PUBLIC_URL}/capmetro-logo.png`}
            alt="CapMetro logo"
            className="capmetro-logo"
          />
        </a>
      </header>

      <main className="start-content">
        <p className="start-eyebrow">CapMetro Bikeshare Planning Game</p>

        <h1 className="start-title">Where should the next station go?</h1>

        <p className="start-subtitle">
          Choose a location in Austin and see how well your station performs
          compared to existing bikeshare stations.
        </p>

        <div className="instructions-box">
          <div className="instruction-step step-one">
            <span>01</span>
            <p>Press play to open the map.</p>
          </div>

          <div className="instruction-step step-two">
            <span>02</span>
            <p>Click where you think a new station should be placed.</p>
          </div>

          <div className="instruction-step step-three">
            <span>03</span>
            <p>Submit your location and get a predicted performance score.</p>
          </div>
        </div>

        <button
          ref={playButtonRef}
          className="play-button"
          onClick={handlePlayClick}
          disabled={isTransitioning}
        >
          Play
        </button>
      </main>

      <HintsButton className="start-screen-hints" />

      <a
        className="github-link"
        href="https://github.com/kc4stillo/bikeshare_analysis"
        target="_blank"
        rel="noreferrer"
        aria-label="View project on GitHub"
      >
        <svg
          viewBox="0 0 24 24"
          width="22"
          height="22"
          aria-hidden="true"
          className="github-icon"
        >
          <path
            fill="currentColor"
            d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.79 8.21 11.38.6.11.82-.26.82-.58v-2.17c-3.34.73-4.04-1.42-4.04-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.74.08-.74 1.21.09 1.85 1.25 1.85 1.25 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23.96-.27 1.98-.4 3-.4s2.04.13 3 .4c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.49 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58C20.56 21.79 24 17.3 24 12c0-6.63-5.37-12-12-12z"
          />
        </svg>
        <span>GitHub</span>
      </a>

      {isTransitioning && <span className="map-transition-bubble" />}
    </div>
  );
}