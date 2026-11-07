import PropTypes from "prop-types";

function GradCamPanel({ faceImage, gradcamImage }) {
  if (!gradcamImage) {
    return null;
  }

  return (
    <div className="panel grid gap-5 p-5 lg:grid-cols-2">
      <div>
        <p className="panel-heading">Original Face</p>
        <img
          src={`data:image/png;base64,${faceImage}`}
          alt="Face crop"
          className="mt-4 h-72 w-full rounded-2xl object-cover transition duration-300 hover:scale-[1.01]"
        />
      </div>
      <div>
        <p className="panel-heading">Grad-CAM Overlay</p>
        <img
          src={`data:image/png;base64,${gradcamImage}`}
          alt="GradCAM heatmap"
          className="mt-4 h-72 w-full rounded-2xl object-cover transition duration-300 hover:scale-[1.01]"
        />
      </div>
    </div>
  );
}

GradCamPanel.propTypes = {
  faceImage: PropTypes.string,
  gradcamImage: PropTypes.string,
};

export default GradCamPanel;
