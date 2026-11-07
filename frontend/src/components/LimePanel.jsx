import PropTypes from "prop-types";

function LimePanel({ limeImage, landmarkImage }) {
  if (!limeImage && !landmarkImage) {
    return null;
  }

  return (
    <div className="panel grid gap-5 p-5 lg:grid-cols-2">
      {limeImage ? (
        <div>
          <p className="panel-heading">LIME Segmentation</p>
          <img
            src={`data:image/png;base64,${limeImage}`}
            alt="LIME explanation"
            className="mt-4 h-72 w-full rounded-2xl object-cover transition duration-300 hover:scale-[1.01]"
          />
        </div>
      ) : null}

      {landmarkImage ? (
        <div>
          <p className="panel-heading">Face Landmarks</p>
          <img
            src={`data:image/png;base64,${landmarkImage}`}
            alt="Face landmarks"
            className="mt-4 h-72 w-full rounded-2xl object-cover transition duration-300 hover:scale-[1.01]"
          />
        </div>
      ) : null}
    </div>
  );
}

LimePanel.propTypes = {
  limeImage: PropTypes.string,
  landmarkImage: PropTypes.string,
};

export default LimePanel;
