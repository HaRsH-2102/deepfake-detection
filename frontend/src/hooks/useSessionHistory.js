import { useEffect, useState } from "react";

const STORAGE_KEY = "deepfake_history";

function readHistory() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    return [];
  }
}

export function useSessionHistory() {
  const [history, setHistory] = useState(() => readHistory());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  function addEntry(entry) {
    setHistory((current) => {
      const next = [entry, ...current].slice(0, 10);
      return next;
    });
  }

  function getHistory() {
    return history;
  }

  function clearHistory() {
    setHistory([]);
  }

  return {
    history,
    addEntry,
    getHistory,
    clearHistory,
  };
}
