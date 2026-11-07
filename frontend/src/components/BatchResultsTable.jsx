import { useMemo, useState } from "react";
import PropTypes from "prop-types";

function BatchResultsTable({ results, summary }) {
  const [sortField, setSortField] = useState("confidence");
  const [sortDirection, setSortDirection] = useState("desc");

  const sortedResults = useMemo(() => {
    const cloned = [...results];
    cloned.sort((left, right) => {
      const leftValue = left[sortField];
      const rightValue = right[sortField];
      if (leftValue === rightValue) {
        return 0;
      }
      const comparison = leftValue > rightValue ? 1 : -1;
      return sortDirection === "asc" ? comparison : -comparison;
    });
    return cloned;
  }, [results, sortField, sortDirection]);

  function updateSort(field) {
    if (sortField === field) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortField(field);
    setSortDirection("desc");
  }

  function exportCsv() {
    const rows = [
      ["filename", "prediction", "confidence"],
      ...sortedResults.map((result) => [
        result.filename,
        result.prediction,
        result.confidence,
      ]),
    ];
    const blob = new Blob([rows.map((row) => row.join(",")).join("\n")], {
      type: "text/csv;charset=utf-8;",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "batch-results.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  if (!results?.length) {
    return null;
  }

  return (
    <div className="panel overflow-hidden">
      <div className="flex flex-col gap-4 border-b border-forensic-border px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="panel-heading">Batch Summary</p>
          <div className="mt-3 flex flex-wrap gap-3">
            <span className="data-chip">{summary.total} total</span>
            <span className="data-chip text-forensic-fake">{summary.fake_count} fake</span>
            <span className="data-chip text-forensic-real">{summary.real_count} real</span>
            <span className="data-chip">{(summary.avg_confidence * 100).toFixed(1)}% avg</span>
          </div>
        </div>
        <button
          type="button"
          onClick={exportCsv}
          className="rounded-full border border-forensic-border px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-forensic-text transition hover:border-forensic-accent hover:text-forensic-accent"
        >
          Export CSV
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-black/10 text-forensic-muted">
            <tr>
              {[
                ["filename", "Filename"],
                ["prediction", "Prediction"],
                ["confidence", "Confidence"],
              ].map(([field, label]) => (
                <th
                  key={field}
                  className="cursor-pointer px-5 py-3 font-mono text-xs uppercase tracking-[0.2em]"
                  onClick={() => updateSort(field)}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((result) => (
              <tr key={result.filename} className="border-t border-forensic-border">
                <td className="px-5 py-4 text-forensic-text">{result.filename}</td>
                <td className="px-5 py-4">
                  <span
                    className="rounded-full px-3 py-1 font-mono text-xs"
                    style={{
                      background:
                        result.prediction === "FAKE"
                          ? "rgba(239,68,68,0.12)"
                          : "rgba(34,197,94,0.12)",
                      color:
                        result.prediction === "FAKE"
                          ? "var(--verdict-fake)"
                          : "var(--verdict-real)",
                    }}
                  >
                    {result.prediction}
                  </span>
                </td>
                <td className="px-5 py-4 font-mono text-forensic-text">
                  {(result.confidence * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

BatchResultsTable.propTypes = {
  results: PropTypes.arrayOf(PropTypes.object).isRequired,
  summary: PropTypes.object.isRequired,
};

export default BatchResultsTable;
