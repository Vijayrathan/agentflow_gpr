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
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, generatedFile]);

  // Set generatedFile from messages if it contains a file but generatedFile is not set
  useEffect(() => {
    // Look for a message with file content, starting from the most recent
    const fileMessage = [...messages]
      .reverse()
      .find(
        (msg) =>
          msg.role === "assistant" &&
          (msg.file_content ||
            msg.generated_file_path ||
            (msg.content && parseInputParametersBlock(msg.content)))
      );

    if (fileMessage) {
      const parsed = parseInputParametersBlock(fileMessage.content);
      if (parsed && parsed.code) {
        const newFile = {
          content: parsed.code,
          filename: fileMessage.generated_file_path
            ? fileMessage.generated_file_path.split("/").pop()
            : "generated.in",
          file_path: fileMessage.generated_file_path,
        };
        // Only update if it's different to avoid infinite loops
        if (!generatedFile || generatedFile.content !== newFile.content) {
          setGeneratedFile(newFile);
        }
      } else if (fileMessage.file_content) {
        const newFile = {
          content: fileMessage.file_content,
          filename: fileMessage.generated_file_path
            ? fileMessage.generated_file_path.split("/").pop()
            : "generated.in",
          file_path: fileMessage.generated_file_path,
        };
        // Only update if it's different to avoid infinite loops
        if (!generatedFile || generatedFile.content !== newFile.content) {
          setGeneratedFile(newFile);
        }
      }
    }
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
          generated_file_path: data.generated_file_path,
          file_content: data.file_content,
        };
        setMessages((prev) => [...prev, assistantMessage]);

        // If complete, set the generated file from response or fetch it
        if (data.status === "complete") {
          // Try to extract file content from the message if not in data
          const parsed = parseInputParametersBlock(data.message || "");
          const fileContent =
            data.file_content || (parsed && parsed.code) || null;

          if (fileContent) {
            setGeneratedFile({
              content: fileContent,
              filename: data.generated_file_path
                ? data.generated_file_path.split("/").pop()
                : "generated.in",
              file_path: data.generated_file_path,
            });
          } else if (data.generated_file_path) {
            // Fallback: fetch the file if content not in response
            fetch("/api/file")
              .then((res) => res.json())
              .then((fileData) => {
                if (fileData.content) {
                  setGeneratedFile({
                    content: fileData.content,
                    filename: fileData.filename || "generated.in",
                    file_path: data.generated_file_path,
                  });
                }
              })
              .catch((err) => {
                console.error("Error fetching file:", err);
              });
          } else if (parsed && parsed.code) {
            // If we parsed file from message but no file_path, still set the file
            setGeneratedFile({
              content: parsed.code,
              filename: "generated.in",
              file_path: null, // Will try to find from messages
            });
          }

          // Add a follow-up message asking if user wants to simulate
          // Only add if we haven't already shown simulation prompt and file was generated
          if (data.generated_file_path) {
            // Use a small delay to ensure the file display is rendered first
            setTimeout(() => {
              setMessages((prev) => {
                // Check if we already have a simulation prompt message
                const hasSimulationPrompt = prev.some(
                  (msg) =>
                    msg.role === "assistant" &&
                    msg.content &&
                    (msg.content.includes("Would you like to run") ||
                      msg.content.includes("Run Simulation"))
                );

                if (!hasSimulationPrompt) {
                  return [
                    ...prev,
                    {
                      role: "assistant",
                      content:
                        "The input file has been generated successfully! Would you like to run the simulation? Click the 'Run Simulation' button below the file to start.",
                      status: "complete",
                    },
                  ];
                }
                return prev;
              });
            }, 200);
          }
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
      if (inputRef.current) {
        inputRef.current.style.height = "auto";
        inputRef.current.focus();
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const adjustTextareaHeight = () => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      const scrollHeight = inputRef.current.scrollHeight;
      const maxHeight = window.innerHeight * 0.4; // 40% of viewport height
      inputRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
      inputRef.current.style.overflowY =
        scrollHeight > maxHeight ? "auto" : "hidden";
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [input]);

  useEffect(() => {
    // Initial resize on mount
    adjustTextareaHeight();

    const handleResize = () => {
      adjustTextareaHeight();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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
      setSimulationResult(null);
    } catch (error) {
      console.error("Error resetting conversation:", error);
    }
  };

  const runSimulation = async (filePath) => {
    if (!filePath) {
      console.error("No file path provided for simulation");
      return;
    }

    setSimulationLoading(true);
    setSimulationResult(null);

    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_path: filePath,
          session_id: sessionId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSimulationResult({
          status: "success",
          message: data.message || "Simulation completed successfully",
          result: data.result || "",
        });
        // Add simulation result as a message
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Simulation completed successfully!\n\n${
              data.result || data.message
            }`,
            status: "complete",
          },
        ]);
      } else {
        setSimulationResult({
          status: "error",
          message: data.message || data.error || "Simulation failed",
          error: data.error || "",
        });
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Simulation failed: ${
              data.message || data.error || "Unknown error"
            }`,
            status: "error",
          },
        ]);
      }
    } catch (error) {
      const errorMessage = error.message || "Network error";
      setSimulationResult({
        status: "error",
        message: errorMessage,
        error: errorMessage,
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Simulation error: ${errorMessage}`,
          status: "error",
        },
      ]);
    } finally {
      setSimulationLoading(false);
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

  const downloadContentAsFile = (content, filename = "generated.in") => {
    if (!content) return;
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = async (text, label = "Code") => {
    try {
      await navigator.clipboard.writeText(text);
      // Show a temporary success message
      const notification = document.createElement("div");
      notification.className = "copy-notification";
      notification.textContent = `${label} copied to clipboard!`;
      document.body.appendChild(notification);
      setTimeout(() => {
        notification.classList.add("show");
      }, 10);
      setTimeout(() => {
        notification.classList.remove("show");
        setTimeout(() => {
          document.body.removeChild(notification);
        }, 300);
      }, 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const parseInputParametersBlock = (text) => {
    if (!text) return null;

    const markerLower = "input parameters file:";
    const lower = text.toLowerCase();
    const markerIndex = lower.indexOf(markerLower);
    if (markerIndex === -1) return null;

    // Look for fenced code block after the marker
    const firstFence = text.indexOf("```", markerIndex);
    if (firstFence === -1) return null;
    const secondFence = text.indexOf("```", firstFence + 3);
    if (secondFence === -1) return null;

    const before = text.slice(0, markerIndex).trim();
    const header = text.slice(markerIndex, firstFence).trimEnd();
    const code = text.slice(firstFence + 3, secondFence).trim();
    const after = text.slice(secondFence + 3).trim();

    return { before, header, code, after };
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
            <h1>Intelligent GPR Simulator</h1>
            <p>Generate gprMax input files and simulate them.</p>
          </div>
          <button
            className="reset-button"
            onClick={resetConversation}
            title="Reset Conversation"
            aria-label="Reset conversation and start fresh"
          >
            Reset
          </button>
        </div>

        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                <div className="message-text">
                  {(() => {
                    const parsed =
                      msg.role === "assistant"
                        ? parseInputParametersBlock(msg.content)
                        : null;

                    if (!parsed) {
                      return msg.content.split("\n").map((line, i, arr) => (
                        <React.Fragment key={i}>
                          {line}
                          {i < arr.length - 1 && <br />}
                        </React.Fragment>
                      ));
                    }

                    return (
                      <>
                        {parsed.before && (
                          <div className="message-prefix">
                            {parsed.before.split("\n").map((line, i, arr) => (
                              <React.Fragment key={i}>
                                {line}
                                {i < arr.length - 1 && <br />}
                              </React.Fragment>
                            ))}
                          </div>
                        )}
                        <div className="input-file-block">
                          <div className="input-file-header">
                            <span>{parsed.header}</span>
                            <div className="file-actions">
                              <button
                                className="copy-button"
                                onClick={() =>
                                  copyToClipboard(parsed.code, "Code")
                                }
                                aria-label="Copy code to clipboard"
                                title="Copy code"
                              >
                                Copy
                              </button>
                              <button
                                className="download-button"
                                onClick={() =>
                                  downloadContentAsFile(
                                    parsed.code,
                                    "generated_from_chat.in"
                                  )
                                }
                                aria-label="Download file"
                                title="Download file"
                              >
                                Download
                              </button>
                              {!simulationResult && (
                                <button
                                  className="simulate-button"
                                  onClick={() => {
                                    // Get file path from the message or find it in messages
                                    const filePath =
                                      msg.generated_file_path ||
                                      messages.find(
                                        (m) => m.generated_file_path
                                      )?.generated_file_path;

                                    if (filePath) {
                                      runSimulation(filePath);
                                    } else {
                                      alert(
                                        "File path not available. The file needs to be regenerated to run simulation."
                                      );
                                    }
                                  }}
                                  disabled={simulationLoading}
                                  aria-label="Run simulation"
                                  title="Run simulation"
                                >
                                  {simulationLoading ? (
                                    <span
                                      className="spinner-small"
                                      aria-hidden="true"
                                    ></span>
                                  ) : (
                                    "Run Simulation"
                                  )}
                                </button>
                              )}
                            </div>
                          </div>
                          <pre className="file-content">{parsed.code}</pre>
                        </div>
                        {parsed.after && (
                          <div className="message-suffix">
                            {parsed.after.split("\n").map((line, i, arr) => (
                              <React.Fragment key={i}>
                                {line}
                                {i < arr.length - 1 && <br />}
                              </React.Fragment>
                            ))}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
                {msg.status === "error" && (
                  <div className="status-badge error">Error</div>
                )}
                {msg.thought_process && msg.thought_process.length > 0 && (
                  <div className="thought-process-container">
                    <button
                      className="thought-process-toggle"
                      onClick={() => toggleThoughtProcess(idx)}
                      aria-label={
                        expandedThoughts.has(idx)
                          ? "Collapse thinking process"
                          : "Expand thinking process"
                      }
                      aria-expanded={expandedThoughts.has(idx)}
                    >
                      <span className="thought-process-icon" aria-hidden="true">
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
                <div className="file-actions">
                  <button
                    className="copy-button"
                    onClick={() =>
                      copyToClipboard(generatedFile.content, "File")
                    }
                    aria-label="Copy file content to clipboard"
                    title="Copy file content"
                  >
                    Copy
                  </button>
                  <button
                    className="download-button"
                    onClick={downloadFile}
                    aria-label="Download file"
                    title="Download file"
                  >
                    Download
                  </button>
                </div>
              </div>
              <pre className="file-content">{generatedFile.content}</pre>
              {simulationLoading && (
                <div className="simulation-loading">
                  <span className="thinking-dots">
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                  </span>
                  <span>Running simulation...</span>
                </div>
              )}
              {simulationResult && (
                <div className={`simulation-result ${simulationResult.status}`}>
                  <div className="simulation-result-header">
                    <span className="simulation-status">
                      {simulationResult.status === "success" ? "✓" : "✗"}{" "}
                      Simulation{" "}
                      {simulationResult.status === "success"
                        ? "Completed"
                        : "Failed"}
                    </span>
                  </div>
                  <div className="simulation-result-content">
                    <pre>
                      {simulationResult.result || simulationResult.message}
                    </pre>
                  </div>
                </div>
              )}
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
            onChange={(e) => {
              setInput(e.target.value);
              adjustTextareaHeight();
            }}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            disabled={loading}
            aria-label="Message input"
          />
          <button
            className="send-button"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            {loading ? (
              <span className="spinner" aria-hidden="true"></span>
            ) : (
              <span>Send</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
