import { useEffect, useMemo, useState } from "react";
import PropTypes from "prop-types";
import { useDropzone } from "react-dropzone";

function UploadZone({
  title,
  description,
  accept,
  multiple = false,
  files,
  onFilesChange,
}) {
  const [previewUrl, setPreviewUrl] = useState(null);

  const dropzone = useDropzone({
    accept,
    multiple,
    onDrop: (acceptedFiles) => {
      onFilesChange(multiple ? acceptedFiles : acceptedFiles[0] || null);
    },
  });

  const normalizedFiles = useMemo(() => {
    if (!files) {
      return [];
    }
    return Array.isArray(files) ? files : [files];
  }, [files]);

  useEffect(() => {
    if (!normalizedFiles.length || !normalizedFiles[0]?.type?.startsWith("image/")) {
      setPreviewUrl(null);
      return undefined;
    }

    const nextUrl = URL.createObjectURL(normalizedFiles[0]);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [normalizedFiles]);

  return (
    <div className="panel px-5 py-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="panel-heading">{title}</p>
          <p className="mt-2 text-sm text-forensic-muted">{description}</p>
        </div>
        <div className="data-chip">{multiple ? "MULTI-FILE" : "SINGLE INPUT"}</div>
      </div>

      <div
        {...dropzone.getRootProps()}
        className={`group cursor-pointer rounded-[1.75rem] border border-dashed px-5 py-10 text-center transition ${
          dropzone.isDragActive
            ? "border-forensic-accent bg-forensic-accent/10"
            : "border-forensic-border bg-black/10 hover:border-forensic-accent hover:bg-forensic-accent/5"
        }`}
      >
        <input {...dropzone.getInputProps()} />
        <p className="font-mono text-sm uppercase tracking-[0.24em] text-forensic-text">
          Drag and drop files here
        </p>
        <p className="mt-3 text-sm text-forensic-muted">
          or click to browse supported files
        </p>
        <p className="mt-5 text-xs uppercase tracking-[0.2em] text-forensic-muted">
          {Object.values(accept)
            .flat()
            .join(" | ")}
        </p>
      </div>

      {normalizedFiles.length > 0 && (
        <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="rounded-2xl border border-forensic-border bg-black/10 p-4">
            <p className="panel-heading">Queued Files</p>
            <div className="mt-3 space-y-2">
              {normalizedFiles.map((file) => (
                <div
                  key={`${file.name}-${file.size}`}
                  className="flex items-center justify-between rounded-2xl border border-forensic-border px-3 py-2 text-sm text-forensic-text"
                >
                  <span className="truncate">{file.name}</span>
                  <span className="font-mono text-xs text-forensic-muted">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              ))}
            </div>
          </div>

          {previewUrl && (
            <div className="overflow-hidden rounded-2xl border border-forensic-border bg-black/20 p-2">
              <img
                src={previewUrl}
                alt="Preview"
                className="h-44 w-full rounded-xl object-cover lg:w-56"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

UploadZone.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  accept: PropTypes.object.isRequired,
  multiple: PropTypes.bool,
  files: PropTypes.oneOfType([
    PropTypes.arrayOf(PropTypes.object),
    PropTypes.object,
  ]),
  onFilesChange: PropTypes.func.isRequired,
};

export default UploadZone;
