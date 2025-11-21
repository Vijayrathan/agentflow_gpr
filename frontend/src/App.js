import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(
    () => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );
  const [generatedFile, setGeneratedFile] = useState(null);
  const [expandedThoughts, setExpandedThoughts] = useState(new Set());
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, generatedFile]);

  useEffect(() => {
    // Welcome message
    setMessages([
      {
        role: "assistant",
        content:
          "Welcome! I'm here to help you generate gprMax input files. Tell me about the simulation you want to create, and I'll guide you through the process.\n\nYou can describe your model, layers, waveform, antenna, and other parameters. I'll ask for any missing information until we have everything needed to generate the input file.",
      },
    ]);
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    // Add user message to chat
    const newUserMessage = { role: "user", content: userMessage };
    setMessages((prev) => [...prev, newUserMessage]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      // Check if response has content before parsing JSON
      const contentType = response.headers.get("content-type");
      let data;

      if (contentType && contentType.includes("application/json")) {
        const text = await response.text();
        if (text) {
          try {
            data = JSON.parse(text);
          } catch (parseError) {
            throw new Error(`Invalid JSON response: ${text.substring(0, 100)}`);
          }
        } else {
          throw new Error("Empty response from server");
        }
      } else {
        const text = await response.text();
        throw new Error(`Unexpected response type: ${text.substring(0, 100)}`);
      }

      if (response.ok) {
        const assistantMessage = {
          role: "assistant",
          content: data.message || "Response received",
          status: data.status,
          thought_process: data.thought_process || [],
        };
        setMessages((prev) => [...prev, assistantMessage]);

        // If complete, fetch the generated file
        if (data.status === "complete") {
          fetch("/api/file")
            .then((res) => res.json())
            .then((fileData) => {
              if (fileData.content) {
                setGeneratedFile({
                  content: fileData.content,
                  filename: fileData.filename || "generated.in",
                });
              }
            })
            .catch((err) => {
              console.error("Error fetching file:", err);
            });
        }
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${
              data.message || data.error || "Something went wrong"
            }`,
            status: "error",
          },
        ]);
      }
    } catch (error) {
      let errorMessage = error.message;
      if (
        error.message.includes("Failed to fetch") ||
        error.message.includes("NetworkError")
      ) {
        errorMessage =
          "Cannot connect to backend server. Please make sure the Flask server is running on port 5002.";
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Network error: ${errorMessage}`,
          status: "error",
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetConversation = async () => {
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ session_id: sessionId }),
      });
      setMessages([
        {
          role: "assistant",
          content:
            "Conversation reset. How can I help you create a new gprMax simulation?",
        },
      ]);
      setGeneratedFile(null);
    } catch (error) {
      console.error("Error resetting conversation:", error);
    }
  };

  const downloadFile = () => {
    if (!generatedFile) return;
    const blob = new Blob([generatedFile.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = generatedFile.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleThoughtProcess = (messageIndex) => {
    setExpandedThoughts((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(messageIndex)) {
        newSet.delete(messageIndex);
      } else {
        newSet.add(messageIndex);
      }
      return newSet;
    });
  };

  const formatToolArgs = (args) => {
    if (!args || typeof args !== "object") return "";
    try {
      return JSON.stringify(args, null, 2);
    } catch {
      return String(args);
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <h1>gprMax Generator</h1>
            <p>Intelligent Input File Creation</p>
          </div>
          <button
            className="reset-button"
            onClick={resetConversation}
            title="Reset Conversation"
          >
            Reset
          </button>
        </div>

        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                <div className="message-text">
                  {msg.content.split("\n").map((line, i) => (
                    <React.Fragment key={i}>
                      {line}
                      {i < msg.content.split("\n").length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </div>
                {msg.status === "error" && (
                  <div className="status-badge error">Error</div>
                )}
                {msg.thought_process && msg.thought_process.length > 0 && (
                  <div className="thought-process-container">
                    <button
                      className="thought-process-toggle"
                      onClick={() => toggleThoughtProcess(idx)}
                    >
                      <span className="thought-process-icon">
                        {expandedThoughts.has(idx) ? "▼" : "▶"}
                      </span>
                      <span>Thinking</span>
                    </button>
                    {expandedThoughts.has(idx) && (
                      <div className="thought-process-content">
                        {msg.thought_process.map((step, stepIdx) => (
                          <div key={stepIdx} className="thought-step">
                            {step.type === "message" && (
                              <div className="thought-message">
                                <span className="thought-label">
                                  {step.role === "assistant"
                                    ? "Assistant"
                                    : "User"}
                                  :
                                </span>
                                <div className="thought-text">
                                  {step.content}
                                </div>
                              </div>
                            )}
                            {step.type === "tool_call" && (
                              <div className="thought-tool-call">
                                <span className="thought-label">
                                  🔧 Tool Call:
                                </span>
                                <div className="thought-tool-name">
                                  {step.tool_name}
                                </div>
                                {step.args &&
                                  Object.keys(step.args).length > 0 && (
                                    <pre className="thought-tool-args">
                                      {formatToolArgs(step.args)}
                                    </pre>
                                  )}
                              </div>
                            )}
                            {step.type === "tool_result" && (
                              <div className="thought-tool-result">
                                <span className="thought-label">
                                  📊 Tool Result:
                                </span>
                                <div className="thought-text">
                                  {step.result}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {generatedFile && (
            <div className="file-display">
              <div className="file-header">
                <span className="file-name">{generatedFile.filename}</span>
                <button className="download-button" onClick={downloadFile}>
                  Download
                </button>
              </div>
              <pre className="file-content">{generatedFile.content}</pre>
            </div>
          )}
          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="thinking-indicator">
                  <span className="thinking-dots">
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                  </span>
                  <span className="thinking-text">Thinking</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <textarea
            ref={inputRef}
            className="message-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            rows={1}
            disabled={loading}
          />
          <button
            className="send-button"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            {loading ? <span className="spinner"></span> : <span>Send</span>}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
