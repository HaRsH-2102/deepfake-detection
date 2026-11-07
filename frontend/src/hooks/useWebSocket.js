import { useEffect, useRef, useState } from "react";

export function useWebSocket(url) {
  const socketRef = useRef(null);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState("idle");
  const [latestMessage, setLatestMessage] = useState(null);

  function connect() {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return Promise.resolve(socketRef.current);
    }

    return new Promise((resolve, reject) => {
      setStatus("connecting");
      const socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => {
        setStatus("connected");
        resolve(socket);
      };

      socket.onmessage = (event) => {
        let payload = event.data;
        try {
          payload = JSON.parse(event.data);
        } catch (error) {
          payload = event.data;
        }
        setLatestMessage(payload);
        setMessages((current) => [...current, payload]);
      };

      socket.onerror = (error) => {
        setStatus("error");
        reject(error);
      };

      socket.onclose = () => {
        setStatus("closed");
      };
    });
  }

  function disconnect() {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }

  function sendMessage(message) {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (
      message instanceof ArrayBuffer ||
      message instanceof Blob ||
      message instanceof Uint8Array
    ) {
      socketRef.current.send(message);
      return true;
    }

    socketRef.current.send(JSON.stringify(message));
    return true;
  }

  useEffect(() => {
    return () => disconnect();
  }, []);

  return {
    messages,
    status,
    latestMessage,
    connect,
    disconnect,
    sendMessage,
  };
}
