import PropTypes from "prop-types";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

function VideoTimeline({
  frameConfidences,
  frameIndices,
  fakeFrameIndices = [],
  threshold = 0.5,
  onSelectFrame,
}) {
  const fakeFrameSet = new Set(fakeFrameIndices);
  const data = frameConfidences.map((confidence, index) => ({
    confidence,
    frame: frameIndices?.[index] ?? index + 1,
    isFakeTrigger: fakeFrameSet.has(frameIndices?.[index] ?? index + 1),
  }));

  if (!data.length) {
    return null;
  }

  return (
    <div className="panel p-5">
      <div className="mb-5 flex items-center justify-between">
        <p className="panel-heading">Temporal Consistency Graph</p>
        <div className="flex flex-wrap gap-2">
          <div className="data-chip">click points for frame focus</div>
          <div className="data-chip text-forensic-fake">
            {fakeFrameIndices.length} trigger frames
          </div>
        </div>
      </div>
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <LineChart
            data={data}
            onClick={(state) => {
              if (state?.activePayload?.[0]) {
                onSelectFrame(state.activePayload[0].payload);
              }
            }}
          >
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="frame" stroke="var(--text-secondary)" />
            <YAxis domain={[0, 1]} stroke="var(--text-secondary)" />
            <Tooltip
              contentStyle={{
                background: "var(--bg-card)",
                border: "1px solid var(--border)",
                borderRadius: 16,
              }}
            />
            <ReferenceLine
              y={threshold}
              stroke="var(--verdict-fake)"
              strokeDasharray="6 6"
            />
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="var(--accent-blue)"
              strokeWidth={3}
              dot={(props) => {
                const { cx, cy, payload } = props;
                if (payload?.isFakeTrigger) {
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={6}
                      fill="var(--verdict-fake)"
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  );
                }
                return <circle cx={cx} cy={cy} r={3} fill="var(--accent-blue)" />;
              }}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

VideoTimeline.propTypes = {
  frameConfidences: PropTypes.arrayOf(PropTypes.number).isRequired,
  frameIndices: PropTypes.arrayOf(PropTypes.number),
  fakeFrameIndices: PropTypes.arrayOf(PropTypes.number),
  threshold: PropTypes.number,
  onSelectFrame: PropTypes.func.isRequired,
};

export default VideoTimeline;
