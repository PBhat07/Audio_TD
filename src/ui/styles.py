CUSTOM_CSS = """
/* ============================================================
   AUDIO TRANSCRIPTION STUDIO
   Dark Purple / Modern AI Interface
   ============================================================ */


/* ============================================================
   1. COLOR SYSTEM
   ============================================================ */

:root {
    --bg-main: #0B0718;
    --bg-card: #17102A;
    --bg-card-hover: #21163A;
    --bg-input: #100B1D;
    --bg-transcript: #0F0A1C;

    --border: #34244A;
    --border-light: #46315F;

    --text-primary: #F7F3FF;
    --text-secondary: #C7BDD6;
    --text-muted: #9589A6;

    --purple: #A855F7;
    --purple-hover: #B96CFF;
    --purple-bright: #C084FC;

    --pink: #EC6BFF;
    --teal: #4DD4E8;

    --success: #62D99A;
}


/* ============================================================
   2. GLOBAL GRADIO THEME OVERRIDES
   Keep these at the highest practical CSS scope so Gradio's
   default light surfaces cannot leak into the application.
   ============================================================ */

html,
body,
.gradio-container {
    background: #0B0718 !important;
    color: #F7F3FF !important;
}


/* Gradio CSS variables */

:root,
.gradio-container {
    --body-background-fill: #0B0718 !important;
    --body-background-fill-dark: #0B0718 !important;

    --background-fill-primary: #0B0718 !important;
    --background-fill-primary-dark: #0B0718 !important;

    --background-fill-secondary: #17102A !important;
    --background-fill-secondary-dark: #17102A !important;

    --block-background-fill: #17102A !important;
    --block-background-fill-dark: #17102A !important;

    --block-label-background-fill: #17102A !important;
    --block-label-background-fill-dark: #17102A !important;

    --block-title-background-fill: #17102A !important;
    --block-title-background-fill-dark: #17102A !important;

    --block-border-color: #34244A !important;
    --block-border-color-dark: #34244A !important;

    --input-background-fill: #100B1D !important;
    --input-background-fill-dark: #100B1D !important;

    --input-background-fill-focus: #100B1D !important;
    --input-background-fill-focus-dark: #100B1D !important;

    --input-background-fill-hover: #150D25 !important;
    --input-background-fill-hover-dark: #150D25 !important;

    --input-border-color: #34244A !important;
    --input-border-color-dark: #34244A !important;

    --body-text-color: #F7F3FF !important;
    --body-text-color-dark: #F7F3FF !important;

    --body-text-color-subdued: #C7BDD6 !important;
    --body-text-color-subdued-dark: #C7BDD6 !important;

    --block-label-text-color: #F7F3FF !important;
    --block-label-text-color-dark: #F7F3FF !important;

    --block-title-text-color: #F7F3FF !important;
    --block-title-text-color-dark: #F7F3FF !important;

    --button-primary-background-fill: #A855F7 !important;
    --button-primary-background-fill-hover: #B96CFF !important;
    --button-primary-text-color: #FFFFFF !important;

    --button-secondary-background-fill: #21163A !important;
    --button-secondary-background-fill-hover: #2A1D45 !important;
    --button-secondary-text-color: #F7F3FF !important;

    --border-color-accent-subdued: #34244A !important;

    --loader-color: #A855F7 !important;

    --shadow-drop:
        0 8px 28px rgba(0, 0, 0, 0.28) !important;
}


/* ============================================================
   3. PAGE CONTAINER
   ============================================================ */

.gradio-container {
    width: 100% !important;
    max-width: 1100px !important;
    min-height: 100vh !important;

    margin: 0 auto !important;
    padding: 30px 32px 48px !important;

    background: #0B0718 !important;
    color: #F7F3FF !important;
    box-sizing: border-box !important;
}

body {
    margin: 0 !important;
    background: #0B0718 !important;
    color: #F7F3FF !important;
}


/* ============================================================
   4. GRADIO COMPONENT SURFACES
   Important:
   Do NOT make every Gradio block transparent.
   Give nested Gradio surfaces the same dark card color.
   This prevents the light patches seen in the screenshot.
   ============================================================ */

.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-group {
    background: #17102A !important;
    color: #F7F3FF !important;
    border-color: #34244A !important;
}


/* Markdown/component wrappers inside our cards should not
   introduce another visible surface. */

.input-card .gr-block,
.results-card .gr-block,
.export-card .gr-block,
.input-card .gr-box,
.results-card .gr-box,
.export-card .gr-box,
.input-card .gr-form,
.results-card .gr-form,
.export-card .gr-form,
.input-card .gr-panel,
.results-card .gr-panel,
.export-card .gr-panel {
    background: transparent !important;
    color: #F7F3FF !important;
    border-color: transparent !important;
}


/* ============================================================
   5. APPLICATION CARDS
   ============================================================ */

.input-card,
.results-card,
.export-card {
    width: 100% !important;
    box-sizing: border-box !important;

    padding: 24px !important;
    margin-bottom: 24px !important;

    background: #17102A !important;
    color: #F7F3FF !important;

    border: 1px solid #34244A !important;
    border-radius: 12px !important;

    box-shadow:
        0 8px 28px rgba(0, 0, 0, 0.28) !important;
}

.input-card:hover,
.results-card:hover,
.export-card:hover {
    border-color: #46315F !important;
}


/* ============================================================
   6. TYPOGRAPHY
   ============================================================ */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #FBF9FF !important;
    font-weight: 700 !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.025em;
    margin-bottom: 8px !important;
}

h2 {
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    margin-bottom: 8px !important;
}

p {
    color: #C7BDD6 !important;
    font-size: 0.94rem;
    font-weight: 500;
    line-height: 1.55;
}


/* Markdown itself */

.prose,
.prose p,
.prose li,
.prose h1,
.prose h2,
.prose h3 {
    background: transparent !important;
}

.prose {
    color: #C7BDD6 !important;
}

.prose p,
.prose li {
    color: #C7BDD6 !important;
    font-weight: 500;
}

.prose strong {
    color: #F7F3FF !important;
    font-weight: 700 !important;
}

.prose h1,
.prose h2,
.prose h3 {
    color: #FBF9FF !important;
}


/* ============================================================
   7. LABELS
   ============================================================ */

label {
    color: #F7F3FF !important;
    font-weight: 650 !important;
}


/* ============================================================
   8. FILE UPLOAD
   Explicitly target every upload state.
   ============================================================ */

.audio-upload {
    width: 100% !important;
    margin-top: 10px !important;
    margin-bottom: 22px !important;
}


/* Upload component outer surfaces */

.audio-upload,
.audio-upload > div,
.audio-upload .wrap,
.audio-upload .container,
.audio-upload .upload-container,
.audio-upload .upload-wrap,
.audio-upload .file-upload,
.audio-upload [data-testid="file-upload"] {
    background: #100B1D !important;
    color: #F7F3FF !important;
    border-color: #563B73 !important;
}


/* Main upload/drop zone */

.audio-upload .upload-container,
.audio-upload .upload-wrap {
    min-height: 130px !important;

    background: #100B1D !important;

    border: 1px dashed #563B73 !important;
    border-radius: 10px !important;
}


/* Hover */

.audio-upload .upload-container:hover,
.audio-upload .upload-wrap:hover {
    background: #150D25 !important;
    border-color: #A855F7 !important;
}


/* Upload text */

.audio-upload span,
.audio-upload p {
    color: #C7BDD6 !important;
    font-weight: 500 !important;
}


/* Upload icon */

.audio-upload svg {
    color: #C084FC !important;
}


/* ============================================================
   9. UPLOADED FILE STATE
   ============================================================ */

.audio-upload .file-preview,
.audio-upload .file-preview > div,
.audio-upload [data-testid="file-preview"] {
    background: #100B1D !important;
    color: #F7F3FF !important;
    border-color: #34244A !important;
}

.audio-upload .file-preview * {
    color: #C7BDD6 !important;
}


/* ============================================================
   10. SPEAKER SETTINGS
   ============================================================ */

.speaker-settings {
    width: 100% !important;
    gap: 14px !important;

    margin-top: 4px !important;
    margin-bottom: 20px !important;

    background: transparent !important;
}


/* Remove nested panel surfaces */

.speaker-settings > div,
.speaker-settings .form,
.speaker-settings .gr-box,
.speaker-settings .gr-form {
    background: transparent !important;
    border-color: transparent !important;
}

.speaker-settings label {
    color: #F7F3FF !important;
    font-weight: 650 !important;
}

.speaker-settings input {
    background: #100B1D !important;
    color: #F7F3FF !important;

    border: 1px solid #34244A !important;
    border-radius: 8px !important;

    font-weight: 600 !important;
}

.speaker-settings input:focus {
    border-color: #A855F7 !important;

    box-shadow:
        0 0 0 2px rgba(168, 85, 247, 0.18) !important;
}


/* ============================================================
   11. GENERAL INPUTS
   ============================================================ */

input,
textarea,
select {
    background: #100B1D !important;
    color: #F7F3FF !important;
    border-color: #34244A !important;
}

input::placeholder,
textarea::placeholder {
    color: #9589A6 !important;
}


/* ============================================================
   12. TRANSCRIBE BUTTON
   ============================================================ */

.transcribe-button {
    width: 100% !important;
    min-height: 48px !important;

    background: #A855F7 !important;
    color: #FFFFFF !important;

    border: none !important;
    border-radius: 8px !important;

    font-weight: 750 !important;
    font-size: 0.96rem !important;
    letter-spacing: 0.01em;

    transition:
        background 0.15s ease,
        transform 0.1s ease;
}

.transcribe-button:hover {
    background: #B96CFF !important;
}

.transcribe-button:active {
    transform: translateY(1px);
}


/* ============================================================
   13. STATUS / PROCESSING
   ============================================================ */

.processing-status {
    width: 100% !important;
    margin-top: 16px !important;

    padding: 12px 14px !important;
    box-sizing: border-box !important;

    background: rgba(168, 85, 247, 0.08) !important;

    border: 1px solid rgba(168, 85, 247, 0.22) !important;
    border-left: 3px solid #A855F7 !important;

    border-radius: 7px !important;
    color: #C7BDD6 !important;
}

.processing-status .prose,
.processing-status p {
    background: transparent !important;
    color: #C7BDD6 !important;
}

.processing-status strong {
    color: #F7F3FF !important;
    font-weight: 700 !important;
}


/* Gradio processing/update surfaces */

.processing-status,
.processing-status > div,
.processing-status .wrap,
[data-testid="block-info"] {
    color: #F7F3FF !important;
}


/* ============================================================
   14. AUDIO PLAYER
   ============================================================ */

.audio-player {
    width: 100% !important;
    margin-top: 8px !important;
}

.audio-player,
.audio-player > div,
.audio-player .wrap {
    background: #100B1D !important;
    border-color: #34244A !important;
    border-radius: 9px !important;
}


/* ============================================================
   15. TRANSCRIPT SEARCH
   ============================================================ */

.transcript-search {
    width: 100% !important;

    margin-top: 10px !important;
    margin-bottom: 16px !important;
}

.transcript-search input,
.transcript-search textarea {
    width: 100% !important;
    min-height: 44px !important;
    box-sizing: border-box !important;

    background: #100B1D !important;
    color: #F7F3FF !important;

    border: 1px solid #34244A !important;
    border-radius: 8px !important;

    font-size: 0.94rem !important;
    font-weight: 550 !important;
}

.transcript-search input::placeholder,
.transcript-search textarea::placeholder {
    color: #9589A6 !important;
}

.transcript-search input:focus,
.transcript-search textarea:focus {
    border-color: #A855F7 !important;

    box-shadow:
        0 0 0 2px rgba(168, 85, 247, 0.16) !important;
}


/* ============================================================
   16. SPEAKER HELP
   ============================================================ */

.speaker-help {
    color: #C7BDD6 !important;
    font-size: 0.88rem;
    font-weight: 500;
}

.speaker-help strong {
    color: #4DD4E8 !important;
    font-weight: 700;
}


/* ============================================================
   17. TRANSCRIPT
   ============================================================ */

.transcript-output {
    width: 100% !important;
    margin-top: 8px !important;

    background: transparent !important;
}

.transcript-container {
    width: 100% !important;
    max-height: 620px;

    overflow-y: auto;
    box-sizing: border-box;

    padding: 6px 8px 6px 4px;

    background: #0F0A1C !important;

    border: 1px solid #34244A;
    border-radius: 10px;
}


/* ============================================================
   18. TRANSCRIPT SEGMENTS
   ============================================================ */

.transcript-segment {
    padding: 15px 17px;
    margin-bottom: 10px;

    background: #17102A !important;

    border: 1px solid #34244A;
    border-left: 4px solid #A855F7;

    border-radius: 8px;

    cursor: pointer;

    transition:
        background 0.12s ease,
        border-color 0.12s ease;
}

.transcript-segment:last-child {
    margin-bottom: 0;
}

.transcript-segment:hover {
    background: #21163A !important;
    border-color: #46315F;
}


/* ============================================================
   19. SPEAKER COLOR GROUPING
   ============================================================ */

.transcript-segment.speaker-0 {
    border-left-color: #A855F7;
}

.transcript-segment.speaker-1 {
    border-left-color: #4DD4E8;
}

.transcript-segment.speaker-2 {
    border-left-color: #C084FC;
}

.transcript-segment.speaker-3 {
    border-left-color: #EC6BFF;
}

.transcript-segment.speaker-4 {
    border-left-color: #7DD3FC;
}

.transcript-segment.speaker-5 {
    border-left-color: #F0ABFC;
}


/* ============================================================
   20. TRANSCRIPT HEADER
   ============================================================ */

.transcript-segment-header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    gap: 16px;
    margin-bottom: 7px;
}


/* ============================================================
   21. SPEAKER NAME + RENAME ICON
   ============================================================ */

.transcript-speaker {
    color: #C084FC !important;

    font-weight: 700;
    font-size: 0.92rem;

    cursor: pointer;
    user-select: none;
}

.transcript-speaker::after {
    content: "  ✎";
    color: #4DD4E8;
    font-size: 0.85rem;
}

.transcript-speaker:hover {
    color: #D6A4FF !important;
}


/* ============================================================
   22. TIMESTAMP
   ============================================================ */

.transcript-timestamp {
    color: #B9ADC9;

    font-size: 0.8rem;
    font-weight: 550;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        monospace;

    white-space: nowrap;
}


/* ============================================================
   23. TRANSCRIPT TEXT
   ============================================================ */

.transcript-text {
    color: #F7F3FF !important;

    font-size: 1rem;
    font-weight: 550;

    line-height: 1.65;
    letter-spacing: 0.005em;
}


/* ============================================================
   24. EMPTY TRANSCRIPT
   ============================================================ */

.transcript-empty {
    width: 100% !important;
    box-sizing: border-box !important;

    padding: 54px 24px;
    text-align: center;

    background: #0F0A1C !important;

    border: 1px dashed #34244A;
    border-radius: 10px;
}

.transcript-empty p {
    margin: 0;
    color: #A99CB9 !important;
    font-weight: 550;
}


/* ============================================================
   25. EXPORT / DOWNLOAD
   ============================================================ */

.export-card {
    margin-bottom: 0 !important;
}


/* Main download component */

.download-json {
    width: 100% !important;
    margin-top: 12px !important;

    background: #100B1D !important;
    color: #F7F3FF !important;

    border: 1px solid #34244A !important;
    border-radius: 9px !important;

    box-sizing: border-box !important;
}


/* Remove Gradio's default light surfaces */

.download-json > div,
.download-json .wrap,
.download-json .container,
.download-json .file-preview,
.download-json .file-preview > div,
.download-json [data-testid="file-preview"],
.download-json [data-testid="file"] {
    background: #100B1D !important;
    color: #F7F3FF !important;

    border-color: #34244A !important;
    border-radius: 8px !important;
}


/* File name */

.download-json .file-preview-name,
.download-json .file-name,
.download-json [data-testid="file-name"] {
    color: #F7F3FF !important;
    font-weight: 650 !important;
}


/* All text inside the download area */

.download-json span,
.download-json p,
.download-json div {
    color: #F7F3FF !important;
}


/* ============================================================
   DOWNLOAD ICON
   ============================================================ */

/* Gradio download SVG/icon */

.download-json svg {
    width: 20px !important;
    height: 20px !important;

    color: #C084FC !important;
    fill: currentColor !important;
    stroke: currentColor !important;

    opacity: 1 !important;
}


/* Make the icon particularly visible on hover */

.download-json:hover svg {
    color: #D6A4FF !important;
}


/* ============================================================
   DOWNLOAD BUTTON / LINK
   ============================================================ */

.download-json button,
.download-json a {
    background: #21163A !important;
    color: #F7F3FF !important;

    border: 1px solid #46315F !important;
    border-radius: 7px !important;

    font-weight: 700 !important;
    cursor: pointer !important;
}


/* Hover */

.download-json button:hover,
.download-json a:hover {
    background: #2A1D45 !important;
    color: #FFFFFF !important;

    border-color: #A855F7 !important;
}


/* Hover icon */

.download-json button:hover svg,
.download-json a:hover svg {
    color: #D6A4FF !important;
}


/* Focus */

.download-json button:focus,
.download-json a:focus {
    outline: none !important;

    border-color: #A855F7 !important;

    box-shadow:
        0 0 0 2px rgba(168, 85, 247, 0.18) !important;
}


/* ============================================================
   PREVENT WHITE PATCHES FROM GRADIO FILE STATES
   ============================================================ */

.download-json *,
.download-json *::before,
.download-json *::after {
    box-sizing: border-box !important;
}


/* Explicitly kill possible light backgrounds */

.download-json,
.download-json > *,
.download-json > * > *,
.download-json .file-preview,
.download-json .file-preview > *,
.download-json .wrap,
.download-json .wrap > * {
    background-color: #100B1D !important;
}


/* Keep the export card itself dark */

.export-card .download-json {
    background-color: #100B1D !important;
}

/* ============================================================
   26. LOADING / UPDATE STATES
   ============================================================ */

.loading,
.loading-container,
.progress-bar,
.progress-bar-wrap {
    background: #17102A !important;
    color: #F7F3FF !important;
}


/* Prevent component updates from creating white surfaces */

.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="block-info"] > div {
    background: #17102A !important;
    color: #F7F3FF !important;
}


/* ============================================================
   27. BUTTONS
   ============================================================ */

button {
    border-radius: 8px !important;
}


/* ============================================================
   28. SCROLLBAR
   ============================================================ */

.transcript-container::-webkit-scrollbar {
    width: 8px;
}

.transcript-container::-webkit-scrollbar-track {
    background: #0A0614;
    border-radius: 4px;
}

.transcript-container::-webkit-scrollbar-thumb {
    background: #49345F;
    border-radius: 4px;
}

.transcript-container::-webkit-scrollbar-thumb:hover {
    background: #65457F;
}


/* ============================================================
   29. RESPONSIVE
   ============================================================ */

@media (max-width: 700px) {

    .gradio-container {
        padding: 20px 14px 32px !important;
    }

    .input-card,
    .results-card,
    .export-card {
        padding: 18px !important;
    }

    .speaker-settings {
        flex-direction: column !important;
        gap: 8px !important;
    }

    .transcript-segment {
        padding: 13px;
    }

    .transcript-segment-header {
        align-items: flex-start;
    }

    .transcript-text {
        font-size: 0.96rem;
    }
}


/* ============================================================
   30. FINAL DARK-SURFACE OVERRIDES

   Gradio renders Markdown/HTML through several nested wrapper
   elements. Some of those wrappers can keep the default light
   theme background even when the inner content is transparent.
   These rules explicitly remove those light surfaces.
   ============================================================ */

/* Tell the browser and native controls to use dark rendering. */
html,
body,
.gradio-container {
    color-scheme: dark !important;
}

/* The three application cards remain the visual surface. */
.input-card,
.results-card,
.export-card {
    background: #17102A !important;
    color: #F7F3FF !important;
}

/* Structural wrappers directly inside our cards must not become
   independent light rectangles. */
.input-card > div,
.results-card > div,
.export-card > div {
    background: transparent !important;
    color: #F7F3FF !important;
}

/* Markdown component wrappers. */
.gradio-container .gr-markdown,
.gradio-container .gr-markdown > div,
.gradio-container .gr-markdown .prose,
.gradio-container [data-testid="markdown"],
.gradio-container [data-testid="markdown"] > div {
    background: transparent !important;
    background-color: transparent !important;
    color: #F7F3FF !important;
}

/* Markdown content itself. */
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown li,
.gradio-container .gr-markdown h1,
.gradio-container .gr-markdown h2,
.gradio-container .gr-markdown h3,
.gradio-container .gr-markdown h4,
.gradio-container .gr-markdown h5,
.gradio-container .gr-markdown h6,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose h5,
.gradio-container .prose h6 {
    background: transparent !important;
    background-color: transparent !important;
}

/* Only Markdown inside our cards: don't allow Gradio's light
   component surface to appear behind headings/descriptions. */
.input-card .gr-markdown,
.results-card .gr-markdown,
.export-card .gr-markdown,
.input-card [data-testid="markdown"],
.results-card [data-testid="markdown"],
.export-card [data-testid="markdown"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* HTML component wrappers. */
.gradio-container .gr-html,
.gradio-container .gr-html > div,
.gradio-container .gr-html-container,
.gradio-container [data-testid="html"],
.gradio-container [data-testid="html"] > div {
    background: transparent !important;
    background-color: transparent !important;
    color: #F7F3FF !important;
}

/* Transcript HTML has its own dark content container; only its
   Gradio wrappers should be transparent. */
.results-card .transcript-output,
.results-card .transcript-output > div,
.results-card .transcript-output .wrap,
.results-card .transcript-output .gr-html,
.results-card .transcript-output .gr-html-container {
    background: transparent !important;
    background-color: transparent !important;
    color: #F7F3FF !important;
}

/* Keep the actual transcript content dark. */
.results-card .transcript-container,
.results-card .transcript-empty,
.results-card .transcript-segment {
    background: #0F0A1C !important;
}

.results-card .transcript-segment {
    background: #17102A !important;
}

/* Search textbox must remain an input surface, not transparent. */
.results-card .transcript-search,
.results-card .transcript-search > div,
.results-card .transcript-search input,
.results-card .transcript-search textarea {
    background: #100B1D !important;
    color: #F7F3FF !important;
}

/* Audio player must remain dark as well. */
.results-card .audio-player,
.results-card .audio-player > div,
.results-card .audio-player .wrap {
    background: #100B1D !important;
    color: #F7F3FF !important;
}

/* Remove default light backgrounds from component wrappers while
   preserving the explicitly styled controls above. */
.input-card .gr-box,
.input-card .gr-form,
.results-card .gr-box,
.results-card .gr-form,
.export-card .gr-box,
.export-card .gr-form {
    background: transparent !important;
}

/* Export file control remains a dark input-like surface. */
.export-card .download-json,
.export-card .download-json > div,
.export-card .download-json .file-preview {
    background: #100B1D !important;
    color: #F7F3FF !important;
}

"""
