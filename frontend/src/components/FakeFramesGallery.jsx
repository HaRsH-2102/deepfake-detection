import PropTypes from "prop-types";
import { motion } from "framer-motion";

function FakeFramesGallery({ frames }) {
  if (!frames?.length) {
    return null;
  }

  return (
    <div className="grid gap-5 xl:grid-cols-3">
      {frames.map((frame, index) => (
        <motion.article
          key={`${frame.frame_index}-${index}`}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.08 }}
          className="panel overflow-hidden"
        >
          <div className="grid gap-0">
            <img
              src={`data:image/png;base64,${frame.frame_image}`}
              alt={`Frame ${frame.frame_index}`}
              className="h-52 w-full object-cover"
            />
            {frame.gradcam_image ? (
              <img
                src={`data:image/png;base64,${frame.gradcam_image}`}
                alt={`GradCAM ${frame.frame_index}`}
                className="h-52 w-full object-cover"
              />
            ) : null}
          </div>
          <div className="space-y-2 px-5 py-4">
            <div className="flex items-center justify-between">
              <p className="font-mono text-sm uppercase tracking-[0.18em] text-forensic-muted">
                Trigger Frame {frame.frame_index}
              </p>
              <span className="rounded-full bg-forensic-fake/15 px-3 py-1 font-mono text-xs text-forensic-fake">
                {(frame.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-sm text-forensic-muted">
              This frame crossed the fake threshold and contributed directly to the video verdict.
            </p>
          </div>
        </motion.article>
      ))}
    </div>
  );
}

FakeFramesGallery.propTypes = {
  frames: PropTypes.arrayOf(PropTypes.object),
};

export default FakeFramesGallery;
