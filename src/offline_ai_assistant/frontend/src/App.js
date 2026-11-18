
import React, { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000/api/v1";

function App() {
  // Navigation
  const [activePage, setActivePage] = useState("chat");

  // Chat
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Models
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");

  // Benchmark
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkReport, setBenchmarkReport] = useState(null);
  const [benchmarkReports, setBenchmarkReports] = useState([]);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);

  useEffect(() => {
    loadModels();
    loadConversations();
  }, []);

  useEffect(() => {
    if (activePage === "benchmark") {
      loadBenchmarkReports();
    }
  }, [activePage]);

  // ---------------- MODELS ----------------

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API}/models`);
      setModels(response.data.models || []);
      setSelectedModel(response.data.default_model || "");
    } catch (err) {
      setError("Unable to connect to the backend. Check that FastAPI is running.");
    }
  };

  // ---------------- CHAT ----------------

  const loadConversations = async () => {
    try {
      const response = await axios.get(`${API}/conversations`);
      setConversations(response.data);
    } catch (err) {
      console.error("Could not load conversations:", err);
    }
  };

  // Token-by-token streaming chat
  const sendMessage = async (event) => {
    event.preventDefault();

    const question = message.trim();
    if (!question || loading) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    setMessage("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
          conversation_id: conversationId,
          model_name: selectedModel || null,
        }),
      });

      if (!response.ok) {
        let detail = "Message could not be sent. Please check the backend.";

        try {
          const errorData = await response.json();
          detail = errorData.detail || detail;
        } catch {
          // Keep the fallback message.
        }

        throw new Error(detail);
      }

      if (!response.body) {
        throw new Error("Streaming is not supported by this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      const handleStreamEvent = (streamEvent) => {
        if (streamEvent.type === "start") {
          setConversationId(streamEvent.conversation_id);
          return;
        }

        if (streamEvent.type === "token") {
          setMessages((previous) => {
            if (previous.length === 0) return previous;

            const updated = [...previous];
            const lastIndex = updated.length - 1;
            const lastMessage = updated[lastIndex];

            if (lastMessage.role === "assistant") {
              updated[lastIndex] = {
                ...lastMessage,
                content: lastMessage.content + (streamEvent.content || ""),
              };
            }

            return updated;
          });
          return;
        }

        if (streamEvent.type === "done") {
          completed = true;
          setConversationId(streamEvent.conversation_id);

          // Replace streamed tokens with the final validated response.
          setMessages((previous) => {
            if (previous.length === 0) return previous;

            const updated = [...previous];
            const lastIndex = updated.length - 1;
            const lastMessage = updated[lastIndex];

            if (lastMessage.role === "assistant") {
              updated[lastIndex] = {
                ...lastMessage,
                content: streamEvent.response || "",
                mode: streamEvent.mode,
                sources: streamEvent.sources || [],
                model: streamEvent.model,
              };
            }

            return updated;
          });

          loadConversations();
          return;
        }

        if (streamEvent.type === "error") {
          throw new Error(
            streamEvent.detail || "The assistant could not complete the response."
          );
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          handleStreamEvent(JSON.parse(line));
        }
      }

      buffer += decoder.decode();

      if (buffer.trim()) {
        handleStreamEvent(JSON.parse(buffer));
      }

      if (!completed) {
        throw new Error(
          "The stream ended before the assistant completed its response."
        );
      }
    } catch (err) {
      setError(
        err.message ||
          "Message could not be sent. Please check the backend."
      );

      // Remove an empty assistant placeholder if no response was streamed.
      setMessages((previous) => {
        if (previous.length === 0) return previous;

        const lastMessage = previous[previous.length - 1];

        if (
          lastMessage.role === "assistant" &&
          !lastMessage.content
        ) {
          return previous.slice(0, -1);
        }

        return previous;
      });
    } finally {
      setLoading(false);
    }
  };

  const openConversation = async (id) => {
    try {
      const response = await axios.get(`${API}/chat/${id}/history`);
      setConversationId(id);
      setMessages(response.data.messages || []);
      setError("");
      setActivePage("chat");
    } catch (err) {
      setError("Unable to load this conversation.");
    }
  };

  const newConversation = () => {
    setConversationId(null);
    setMessages([]);
    setMessage("");
    setError("");
    setActivePage("chat");
  };

  // ---------------- BENCHMARK ----------------

  const loadBenchmarkReports = async () => {
    try {
      const response = await axios.get(`${API}/benchmark/reports`);
      setBenchmarkReports(response.data || []);
    } catch (err) {
      setBenchmarkError("Unable to load benchmark reports.");
    }
  };

  const runBenchmark = async () => {
    if (!selectedModel || benchmarkLoading) return;

    setBenchmarkLoading(true);
    setBenchmarkError("");
    setBenchmarkReport(null);

    try {
      const response = await axios.post(`${API}/benchmark/run`, {
        model_name: selectedModel,
      });

      setBenchmarkReport(response.data);
      await loadBenchmarkReports();
    } catch (err) {
      setBenchmarkError(
        err.response?.data?.detail ||
          "Benchmark execution failed. Check the backend."
      );
    } finally {
      setBenchmarkLoading(false);
    }
  };

  const openBenchmarkReport = async (id) => {
    setReportLoading(true);
    setBenchmarkError("");

    try {
      const response = await axios.get(
        `${API}/benchmark/reports/${encodeURIComponent(id)}`
      );
      setBenchmarkReport(response.data);
    } catch (err) {
      setBenchmarkError("Unable to open this benchmark report.");
    } finally {
      setReportLoading(false);
    }
  };

  // ---------------- HELPERS ----------------

  const formatMetric = (value, suffix = "") => {
    if (value === undefined || value === null) return "—";
    return `${Number(value).toFixed(2)}${suffix}`;
  };

  const renderBenchmarkResults = () => {
    if (!benchmarkReport) return null;

    return (
      <div className="benchmark-results">
        <div className="benchmark-section-heading">
          <div>
            <h2>Benchmark Results</h2>
            <p>
              Model: {benchmarkReport.model} ·{" "}
              {benchmarkReport.benchmark_id}
            </p>
          </div>
          <span className="benchmark-success">
            {benchmarkReport.successful_tests}/{benchmarkReport.total_tests} Completed
          </span>
        </div>

        <div className="benchmark-cards">
          <div className="benchmark-card">
            <span>Total Tests</span>
            <strong>{benchmarkReport.total_tests}</strong>
          </div>

          <div className="benchmark-card">
            <span>Similarity Score</span>
            <strong>
              {formatMetric(benchmarkReport.average_similarity_score, "%")}
            </strong>
          </div>

          <div className="benchmark-card">
            <span>Response Time</span>
            <strong>
              {formatMetric(
                benchmarkReport.average_metrics?.response_time,
                " sec"
              )}
            </strong>
          </div>

          <div className="benchmark-card">
            <span>Tokens / Second</span>
            <strong>
              {formatMetric(
                benchmarkReport.average_metrics?.tokens_per_second
              )}
            </strong>
          </div>

          <div className="benchmark-card">
            <span>Generated Tokens</span>
            <strong>
              {formatMetric(
                benchmarkReport.average_metrics?.generated_tokens
              )}
            </strong>
          </div>

          <div className="benchmark-card">
            <span>Failed Tests</span>
            <strong>{benchmarkReport.failed_tests}</strong>
          </div>
        </div>

        <div className="benchmark-detail-panel">
          <h3>Performance Details</h3>

          <div className="benchmark-metric-row">
            <span>Model loading time</span>
            <strong>
              {formatMetric(
                benchmarkReport.average_metrics?.model_load_time,
                " sec"
              )}
            </strong>
          </div>

          <div className="benchmark-metric-row">
            <span>Prompt tokens</span>
            <strong>
              {formatMetric(benchmarkReport.average_metrics?.prompt_tokens)}
            </strong>
          </div>

          <div className="benchmark-metric-row">
            <span>Total generation duration</span>
            <strong>
              {formatMetric(
                benchmarkReport.average_metrics?.total_duration,
                " sec"
              )}
            </strong>
          </div>
        </div>

        <div className="benchmark-detail-panel">
          <h3>Individual Test Results</h3>

          <div className="benchmark-table-wrapper">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Test</th>
                  <th>Category</th>
                  <th>Similarity</th>
                  <th>Quality</th>
                  <th>Time</th>
                </tr>
              </thead>

              <tbody>
                {(benchmarkReport.results || []).map((result) => (
                  <tr key={result.test_id}>
                    <td>{result.test_id}</td>
                    <td>{result.category}</td>
                    <td>
                      {formatMetric(
                        result.evaluation?.similarity_score,
                        "%"
                      )}
                    </td>
                    <td>
                      <span
                        className={`quality-label ${
                          result.evaluation?.quality?.toLowerCase() || "unknown"
                        }`}
                      >
                        {result.evaluation?.quality || "N/A"}
                      </span>
                    </td>
                    <td>
                      {formatMetric(result.metrics?.response_time, " sec")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="benchmark-note">
            Similarity is a TF-IDF text-overlap score. It does not independently
            verify factual correctness.
          </p>
        </div>
      </div>
    );
  };

  // ---------------- APPLICATION UI ----------------

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">AI</div>
          <div>
            <h2>Offline AI</h2>
            <span>AI Assistant Platform</span>
          </div>
        </div>

        <div className="sidebar-heading">WORKSPACE</div>

        <button
          className={`nav-item ${activePage === "chat" ? "active" : ""}`}
          onClick={() => setActivePage("chat")}
        >
          <span>💬</span>
          Chat Assistant
        </button>

        <button
          className={`nav-item ${activePage === "benchmark" ? "active" : ""}`}
          onClick={() => setActivePage("benchmark")}
        >
          <span>📊</span>
          Benchmark
        </button>

        {activePage === "chat" && (
          <>
            <button className="new-chat" onClick={newConversation}>
              + New conversation
            </button>

            <div className="sidebar-heading">RECENT CONVERSATIONS</div>

            <div className="conversation-list">
              {conversations.map((conversation) => (
                <button
                  key={conversation.id}
                  className={`conversation-item ${
                    conversationId === conversation.id ? "active" : ""
                  }`}
                  onClick={() => openConversation(conversation.id)}
                >
                  {conversation.title || "New conversation"}
                </button>
              ))}
            </div>
          </>
        )}

        <div className="sidebar-footer">
          <span className="status-dot"></span>
          Running locally
        </div>
      </aside>

      <main className="main">
        {activePage === "chat" ? (
          <>
            <header className="topbar">
              <div>
                <h3>HR Policy Assistant</h3>
                <p>Ask questions about company policies and benefits</p>
              </div>

              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={models.length === 0}
              >
                {models.length === 0 && (
                  <option value="">No models available</option>
                )}

                {models.map((model, index) => (
                  <option
                    key={model.name || index}
                    value={model.name}
                  >
                    {model.name}
                  </option>
                ))}
              </select>
            </header>

            <section className="chat-area">
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-icon">✦</div>
                  <h1>How can I help you today?</h1>
                  <p>
                    Ask me about employee benefits, leave policies, workplace
                    safety, or other HR-related questions.
                  </p>

                  <div className="suggestions">
                    <button
                      onClick={() =>
                        setMessage("What employee benefits are available?")
                      }
                    >
                      Employee benefits
                    </button>

                    <button
                      onClick={() =>
                        setMessage("What is the maternity leave policy?")
                      }
                    >
                      Maternity leave
                    </button>

                    <button
                      onClick={() =>
                        setMessage("How can I raise a grievance?")
                      }
                    >
                      Grievance process
                    </button>
                  </div>
                </div>
              ) : (
                <div className="messages">
                  {messages.map((item, index) => (
                    <div
                      key={item.id || index}
                      className={`message-row ${item.role}`}
                    >
                      <div className="avatar">
                        {item.role === "user" ? "Y" : "AI"}
                      </div>

                      <div className="message-content">
                        <div className="message-author">
                          {item.role === "user" ? "You" : "Assistant"}
                        </div>

                        <div className="message-text">{item.content}</div>

                        {item.mode && (
                          <div className="message-meta">
                            Mode: {item.mode.toUpperCase()}
                            {item.model ? ` · Model: ${item.model}` : ""}
                          </div>
                        )}

                        {item.sources?.length > 0 && (
                          <div className="sources">
                            <strong>Sources</strong>

                            {item.sources.map((source, sourceIndex) => (
                              <div className="source-item" key={sourceIndex}>
                                <span>📄</span>
                                <span>
                                  {typeof source === "string"
                                    ? source
                                    : `${source.file_name || source.filename || "Document"}${
                                        source.page
                                          ? ` — Page ${source.page}`
                                          : ""
                                      }`}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {loading && (
                    <div className="loading-message">
                      Assistant is thinking...
                    </div>
                  )}
                </div>
              )}

              {error && <div className="error-message">{error}</div>}
            </section>

            <form className="input-area" onSubmit={sendMessage}>
              <input
                type="text"
                placeholder="Ask your HR policy question..."
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                disabled={loading}
              />

              <button
                type="submit"
                disabled={loading || !message.trim()}
              >
                {loading ? "..." : "Send"}
              </button>
            </form>

            <footer className="disclaimer">
              Offline AI Assistant · Responses are generated using local AI models
            </footer>
          </>
        ) : (
          <>
            <header className="topbar">
              <div>
                <h3>Model Benchmarking</h3>
                <p>Evaluate and measure local AI model performance</p>
              </div>
            </header>

            <section className="chat-area benchmark-area">
              <div className="benchmark-header">
                <div>
                  <h1>Benchmark Dashboard</h1>
                  <p>
                    Run standardized tests and evaluate your local AI models.
                  </p>
                </div>
              </div>

              <div className="benchmark-control-panel">
                <div className="benchmark-model-control">
                  <label htmlFor="benchmark-model">Select Model</label>

                  <select
                    id="benchmark-model"
                    value={selectedModel}
                    onChange={(event) => setSelectedModel(event.target.value)}
                    disabled={benchmarkLoading || models.length === 0}
                  >
                    {models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  className="run-benchmark-button"
                  onClick={runBenchmark}
                  disabled={benchmarkLoading || !selectedModel}
                >
                  {benchmarkLoading
                    ? "Running Benchmark..."
                    : "▶ Run Benchmark"}
                </button>
              </div>

              {benchmarkLoading && (
                <div className="benchmark-loading">
                  <div className="benchmark-spinner"></div>
                  <strong>Benchmark in progress</strong>
                  <p>
                    Your model is answering 8 test questions. This may take
                    several minutes.
                  </p>
                </div>
              )}

              {(benchmarkError || reportLoading) && (
                <div className="error-message">
                  {reportLoading ? "Loading report..." : benchmarkError}
                </div>
              )}

              {renderBenchmarkResults()}

              <div className="benchmark-history">
                <h2>Previous Benchmark Reports</h2>

                {benchmarkReports.length === 0 ? (
                  <p>No benchmark reports available yet.</p>
                ) : (
                  <div className="benchmark-history-list">
                    {benchmarkReports.map((report) => (
                      <button
                        key={report.benchmark_id}
                        className="benchmark-history-item"
                        onClick={() =>
                          openBenchmarkReport(report.benchmark_id)
                        }
                        disabled={reportLoading}
                      >
                        <div>
                          <strong>{report.model}</strong>
                          <span>{report.benchmark_id}</span>
                        </div>

                        <div className="history-score">
                          {formatMetric(
                            report.average_similarity_score,
                            "%"
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </section>

            <footer className="disclaimer">
              Offline AI Assistant · Local model performance evaluation
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
