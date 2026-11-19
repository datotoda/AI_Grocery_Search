const uploadForm = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const uploadedPreview = document.getElementById("uploaded-preview");
const uploadedImage = document.getElementById("uploaded-image");
const uploadedFilename = document.getElementById("uploaded-filename");
const detectButton = document.getElementById("detect-button");
const detectionStatus = document.getElementById("detection-status");
const objectGrid = document.getElementById("object-grid");
const selectedObject = document.getElementById("selected-object");
const selectedImage = document.getElementById("selected-image");
const selectedCategory = document.getElementById("selected-category");
const selectedConfidence = document.getElementById("selected-confidence");
const searchButton = document.getElementById("search-button");
const searchStatus = document.getElementById("search-status");
const resultsGrid = document.getElementById("results-grid");

let uploadedImageUuid = null;
let selectedCrop = null;

function setStep(number) {
    document.querySelectorAll(".step").forEach((step) => {
        step.classList.toggle("active", Number(step.dataset.step) <= number);
    });
}

function showError(element, message) {
    element.textContent = message;
    element.classList.add("error");
}

function setLoading(button, loading, label) {
    button.disabled = loading;
    button.textContent = loading ? "Please wait..." : label;
}

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    const button = uploadForm.querySelector("button");
    setLoading(button, true, "Upload image");

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Upload failed.");

        uploadedImageUuid = data.image_uuid;
        uploadedImage.src = data.image_url;
        uploadedFilename.textContent = data.filename;
        uploadedPreview.classList.remove("hidden");

        objectGrid.innerHTML = "";
        resultsGrid.innerHTML = "";
        selectedObject.classList.add("hidden");
        detectionStatus.textContent = "Image uploaded. Ready to detect objects.";
        searchStatus.textContent = "Select a detected object to start searching.";
        setStep(1);
    } catch (error) {
        alert(error.message);
    } finally {
        setLoading(button, false, "Upload image");
    }
});

detectButton.addEventListener("click", async () => {
    if (!uploadedImageUuid) return;

    setLoading(detectButton, true, "Detect objects");
    detectionStatus.classList.remove("error");
    detectionStatus.textContent = "Running YOLO object detection...";
    objectGrid.innerHTML = "";
    resultsGrid.innerHTML = "";
    selectedObject.classList.add("hidden");

    try {
        const response = await fetch("/detect_objects", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image_uuid: uploadedImageUuid}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Detection failed.");

        if (!data.objects.length) {
            detectionStatus.textContent = "No objects detected. Try another image.";
            setStep(2);
            return;
        }

        detectionStatus.textContent = `${data.count} object(s) detected. Select one crop below.`;
        data.objects.forEach((object) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "object-card";
            card.innerHTML = `
                <img src="${object.image_url}" alt="${object.category}">
                <span class="object-name">${object.category}</span>
                <span class="muted">Confidence: ${(object.confidence * 100).toFixed(1)}%</span>
            `;
            card.addEventListener("click", () => selectObject(object, card));
            objectGrid.appendChild(card);
        });
        setStep(2);
    } catch (error) {
        showError(detectionStatus, error.message);
    } finally {
        setLoading(detectButton, false, "Detect objects");
    }
});

function selectObject(object, card) {
    document.querySelectorAll(".object-card").forEach((item) => item.classList.remove("selected"));
    card.classList.add("selected");

    selectedCrop = object;
    selectedImage.src = object.image_url;
    selectedCategory.textContent = object.category;
    selectedConfidence.textContent = `Confidence: ${(object.confidence * 100).toFixed(1)}%`;
    selectedObject.classList.remove("hidden");
    searchStatus.textContent = "Ready to search.";
    resultsGrid.innerHTML = "";
    setStep(3);
}

searchButton.addEventListener("click", async () => {
    if (!selectedCrop) return;

    setLoading(searchButton, true, "Search similar images");
    searchStatus.classList.remove("error");
    searchStatus.textContent = "Generating image embedding and searching FAISS...";
    resultsGrid.innerHTML = "";

    try {
        const response = await fetch("/search", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image_uuid: selectedCrop.image_uuid}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Search failed.");

        searchStatus.textContent = `${data.count} similar image(s) found.`;
        data.results.forEach((result, index) => {
            const card = document.createElement("article");
            card.className = "result-card";

            const image = document.createElement("img");
            image.src = result.image_url;
            image.alt = result.class_name || "Dataset image";

            const info = document.createElement("div");
            info.className = "result-info";

            const title = document.createElement("strong");
            title.textContent = result.class_name || "Unknown product";

            const score = document.createElement("span");
            score.className = "similarity";
            score.textContent = `Similarity: ${(result.similarity * 100).toFixed(1)}%`;

            const category = document.createElement("span");
            category.className = "muted";
            category.textContent = result.coarse_class_name || "";

            const rank = document.createElement("span");
            rank.className = "rank";
            rank.textContent = `#${index + 1}`;

            info.append(title, score, category);
            card.append(rank, image, info);
            resultsGrid.appendChild(card);
        });
        setStep(3);
    } catch (error) {
        showError(searchStatus, error.message);
    } finally {
        setLoading(searchButton, false, "Search similar images");
    }
});
