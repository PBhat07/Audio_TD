CUSTOM_JS = """
() => {
    window.seekToTranscriptTime = function(seconds) {
        const audioElements = document.querySelectorAll("audio");

        if (!audioElements.length) {
            return;
        }

        const audio = audioElements[audioElements.length - 1];

        audio.currentTime = seconds;
        audio.play();
    };

    window.searchTranscript = function(query) {
        const segments = document.querySelectorAll(
            ".transcript-segment"
        );

        const normalizedQuery = query
            .toLowerCase()
            .trim();

        segments.forEach(segment => {
            const text = segment
                .querySelector(".transcript-text")
                ?.textContent
                .toLowerCase() || "";

            const speaker = segment
                .querySelector(".transcript-speaker")
                ?.textContent
                .toLowerCase() || "";

            if (
                !normalizedQuery ||
                text.includes(normalizedQuery) ||
                speaker.includes(normalizedQuery)
            ) {
                segment.style.display = "";
            } else {
                segment.style.display = "none";
            }
        });
    };

    window.renameSpeaker = function(element) {
        const currentName = element.textContent.trim();

        const newName = window.prompt(
            "Rename speaker:",
            currentName
        );

        if (!newName || !newName.trim()) {
            return;
        }

        const oldName = currentName;
        const newSpeakerName = newName.trim();

        document
            .querySelectorAll(".transcript-speaker")
            .forEach(speakerElement => {
                if (
                    speakerElement.textContent.trim()
                    === oldName
                ) {
                    speakerElement.textContent =
                        newSpeakerName;
                }
            });
    };
}
"""