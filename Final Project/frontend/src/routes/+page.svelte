<script lang="ts">
  import { onMount } from 'svelte';
  import { askProject, enhanceRating, getExamples, getSummary, predictSentiment } from '$lib/api';
  import type { AskResponse, EnhanceResponse, LlmExamples, ProjectSummary, SentimentResponse } from '$lib/types';

  let summary: ProjectSummary | null = null;
  let examples: LlmExamples | null = null;
  let sentiment: SentimentResponse | null = null;
  let enhancement: EnhanceResponse | null = null;
  let askResult: AskResponse | null = null;
  let scoring = false;
  let asking = false;
  let error = '';
  let askError = '';

  let reviewText = 'This Amazon gift card was easy to send, arrived fast, and made a perfect last-minute gift!';
  let actualRating = 5;
  let sentimentWeight = 0.3;
  let question = 'Why is macro-F1 more useful than accuracy for this project?';
  const suggestedQuestions = [
    {
      label: 'Why TF-IDF?',
      prompt: 'Why TF-IDF instead of embeddings for this project?'
    },
    {
      label: 'LR vs NB',
      prompt: 'Why did Logistic Regression beat Naive Bayes on macro-F1?'
    },
    {
      label: 'Recommender result',
      prompt: 'Why did the sentiment-enhanced recommender fail?'
    },
    {
      label: 'Presentation limits',
      prompt: 'What are the strongest limitations to mention in the presentation?'
    }
  ];

  const fmt = new Intl.NumberFormat('en-US');
  const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
  const scoreWidth = (value: number) => `${Math.max(4, Math.min(100, value * 100))}%`;

  async function load() {
    error = '';
    try {
      [summary, examples] = await Promise.all([getSummary(), getExamples()]);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function runSentiment() {
    scoring = true;
    error = '';
    try {
      sentiment = await predictSentiment(reviewText, actualRating);
      enhancement = await enhanceRating(actualRating, sentiment.score, sentimentWeight);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      scoring = false;
    }
  }

  async function runEnhancement() {
    if (!sentiment) return;
    enhancement = await enhanceRating(actualRating, sentiment.score, sentimentWeight);
  }

  async function runAsk() {
    asking = true;
    askError = '';
    askResult = null;
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      askError = 'Ask a project-specific question so the assistant can retrieve evidence.';
      asking = false;
      return;
    }
    try {
      askResult = await askProject(trimmedQuestion);
    } catch (err) {
      askError = err instanceof Error ? err.message : String(err);
    } finally {
      asking = false;
    }
  }

  onMount(async () => {
    await load();
    await Promise.all([runSentiment(), runAsk()]);
  });
</script>

<svelte:head>
  <title>Gift Cards Sentiment Lab</title>
  <meta
    name="description"
    content="Interactive COMP262 final project demo for Amazon Gift Cards sentiment analysis and recommendation scoring."
  />
</svelte:head>

<main class="page">
  <div class="topbar">
    <div>
      <h1 class="title">Gift Cards Sentiment Lab</h1>
      <p class="subtitle">
        Ask grounded questions about the project, compare the notebook models, and test the live review scorer without leaving the same interface.
      </p>
    </div>
    <div class="status"><span class="dot"></span> Grounded assistant + FastAPI demo</div>
  </div>

  {#if error}
    <p class="error">{error}</p>
  {/if}

  {#if summary}
    <div class="metric-row" style="margin-bottom: 16px;">
      <div class="metric">
        <span>Phase 2 clean reviews</span>
        <strong>{fmt.format(summary.datasets[1].clean_rows)}</strong>
      </div>
      <div class="metric">
        <span>Positive share in model subset</span>
        <strong>{pct(summary.phase2_modeling.positive / summary.phase2_modeling.rows)}</strong>
      </div>
      <div class="metric">
        <span>Best macro-F1</span>
        <strong>{Math.max(...summary.model_metrics.map((m) => m.macro_f1)).toFixed(3)}</strong>
      </div>
      <div class="metric">
        <span>Notebook models compared</span>
        <strong>{summary.model_metrics.length}</strong>
      </div>
    </div>
  {/if}

  <div class="grid">
    <section class="stack">
      <div class="panel">
        <h2>Project Answerer</h2>
        <p>
          Answers are grounded in committed project evidence. If a local vLLM endpoint is configured,
          the assistant rewrites retrieved evidence into a cleaner answer and cites the evidence IDs it used.
        </p>
        <div class="button-row" style="margin-bottom: 12px;">
          {#each suggestedQuestions as preset}
            <button
              class="btn secondary"
              on:click={() => {
                question = preset.prompt;
                runAsk();
              }}
            >
              {preset.label}
            </button>
          {/each}
        </div>
        <label class="field">
          <span>Question</span>
          <textarea bind:value={question} style="min-height: 92px;"></textarea>
        </label>
        <div class="button-row">
          <button class="btn" disabled={asking || question.trim().length === 0} on:click={runAsk}>
            {asking ? 'Grounding...' : 'Ask grounded assistant'}
          </button>
          {#if askResult}
            <div class="status inline-status">
              <span class="dot"></span>
              {#if askResult.mode === 'vllm_grounded'}
                Grounded via {askResult.assistant_model}
              {:else}
                Retrieval fallback
              {/if}
            </div>
          {/if}
        </div>
        {#if askError}
          <p class="error local-error">{askError}</p>
        {/if}
        {#if askResult}
          <div class="result">
            <p><strong>Answer:</strong> {askResult.answer}</p>
          </div>
          {#if askResult.citations.length > 0}
            <div class="citation-row">
              {#each askResult.citations as citation}
                <span class="citation-chip">{citation}</span>
              {/each}
            </div>
          {/if}
          {#if askResult.evidence.length > 0}
            <div class="evidence">
              {#each askResult.evidence as hit}
                <div class="evidence-item" class:cited={askResult.citations.includes(hit.id)}>
                  <strong>{hit.title}</strong>
                  <span class="small">match score {hit.score.toFixed(2)} · id {hit.id}</span>
                  <p>{hit.body}</p>
                </div>
              {/each}
            </div>
          {/if}
        {/if}
      </div>

      <div class="panel">
        <h2>Try a Review</h2>
        <p>Use this live scorer for demo interaction. The notebook remains the authority for trained model results.</p>
        <label class="field">
          <span>Review text</span>
          <textarea bind:value={reviewText}></textarea>
        </label>
        <div class="metric-row" style="grid-template-columns: repeat(2, minmax(0, 1fr));">
          <label class="field">
            <span>Actual rating</span>
            <input type="number" min="1" max="5" step="1" bind:value={actualRating} />
          </label>
          <label class="field">
            <span>Sentiment weight: {sentimentWeight.toFixed(2)}</span>
            <input type="range" min="0" max="1" step="0.05" bind:value={sentimentWeight} on:input={runEnhancement} />
          </label>
        </div>
        <div class="button-row">
          <button class="btn" disabled={scoring} on:click={runSentiment}>{scoring ? 'Running...' : 'Score review'}</button>
          <button
            class="btn secondary"
            on:click={() => {
              reviewText = 'The gift card arrived late, the code did not work, and customer service never fixed the problem.';
              actualRating = 1;
              runSentiment();
            }}
          >
            Load negative sample
          </button>
        </div>

        {#if sentiment}
          <div class="result">
            <span class="label {sentiment.label}">{sentiment.label}</span>
            <p><strong>Score:</strong> {sentiment.score.toFixed(3)} · <strong>Confidence:</strong> {pct(sentiment.confidence)}</p>
            <p>{sentiment.rationale.join(' ')}</p>
          </div>
        {/if}
      </div>

      {#if summary}
        <div class="panel">
          <h2>Model Arena</h2>
          <p>Accuracy is shown, but macro-F1 is the professional read because the dataset is heavily positive.</p>
          <table class="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro-F1</th>
                <th>Weighted-F1</th>
              </tr>
            </thead>
            <tbody>
              {#each summary.model_metrics as model}
                <tr>
                  <td>{model.model}</td>
                  <td>{pct(model.accuracy)}</td>
                  <td>
                    <div class="bar"><span style="width: {scoreWidth(model.macro_f1)}"></span></div>
                    <span class="small">{model.macro_f1.toFixed(3)}</span>
                  </td>
                  <td>{model.weighted_f1.toFixed(3)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>

    <aside class="stack">
      {#if enhancement}
        <div class="panel">
          <h2>Recommendation Lab</h2>
          <p>Blend explicit rating with sentiment-derived virtual rating.</p>
          <div class="metric-row" style="grid-template-columns: 1fr;">
            <div class="metric">
              <span>Virtual rating</span>
              <strong>{enhancement.sentiment_virtual_rating.toFixed(2)}</strong>
            </div>
            <div class="metric">
              <span>Enhanced rating</span>
              <strong>{enhancement.enhanced_rating.toFixed(2)}</strong>
            </div>
          </div>
          <p>{enhancement.interpretation}</p>
        </div>
      {/if}

      {#if summary}
        <div class="panel">
          <h2>Recommender Result</h2>
          <table class="table">
            <thead>
              <tr><th>Model</th><th>MAE</th><th>RMSE</th></tr>
            </thead>
            <tbody>
              {#each summary.recommender_metrics as row}
                <tr>
                  <td>{row.model}</td>
                  <td>{row.mae.toFixed(3)}</td>
                  <td>{row.rmse.toFixed(3)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <div class="panel">
          <h2>Dataset Reality</h2>
          <p>{summary.findings[0]}</p>
          <p>{summary.findings[3]}</p>
        </div>
      {/if}

      {#if examples}
        <div class="panel">
          <h2>LLM Desk</h2>
          <p class="small">Summarizer: {examples.summarization_model}</p>
          <p>{examples.summaries[0].summary}</p>
          <p class="small">Service response guardrail:</p>
          <p>{examples.service_response.guarded_response}</p>
        </div>
      {/if}
    </aside>
  </div>
</main>
