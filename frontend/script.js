$(document).ready(function() {
    const uploadPage = $('#upload-page');
    const resultPage = $('#result-page');
    const loadingOverlay = $('#loading-overlay');

    const dropZone = $('#drop-zone');
    const imageUpload = $('#image-upload');
    const imagePreview = $('#image-preview');
    const highlightCanvas = $('#highlight-canvas')[0];
    const ctx = highlightCanvas ? highlightCanvas.getContext('2d') : null;

    const customInsights = $('#custom-insights');
    const hfInsights = $('#hf-insights');
    const resetButton = $('#reset-button');
    const resultVerdictText = $('#result-verdict-text');
    const verdictBox = $('.verdict-box'); // 배경색 변경을 위해

    let barChart;
    let lastCustomResult = null;

    // --- 이벤트 핸들러 ---
    dropZone.on('click', () => imageUpload.click());
    dropZone.on('dragover', (e) => { e.preventDefault(); dropZone.addClass('dragover'); });
    dropZone.on('dragleave', (e) => { e.preventDefault(); dropZone.removeClass('dragover'); });
    dropZone.on('drop', (e) => {
        e.preventDefault();
        dropZone.removeClass('dragover');
        if (e.originalEvent.dataTransfer.files.length) handleFile(e.originalEvent.dataTransfer.files[0]);
    });
    imageUpload.on('change', function() { if (this.files.length) handleFile(this.files[0]); });
    resetButton.on('click', resetUI);

    // --- 함수 로직 ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 선택할 수 있습니다.'); return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.attr('src', e.target.result);
            runBothAnalyses(file);
        }
        reader.readAsDataURL(file);
    }

    function runBothAnalyses(file) {
        loadingOverlay.removeClass('d-none');

        const formData = new FormData();
        formData.append('file', file);

        const customModelPromise = $.ajax({
            url: 'http://127.0.0.1:8000/analyze-with-custom-model/',
            type: 'POST', data: formData, processData: false, contentType: false
        });

        const hfModelPromise = $.ajax({
            url: 'http://127.0.0.1:8000/analyze-with-huggingface-model/',
            type: 'POST', data: formData, processData: false, contentType: false
        });

        Promise.all([customModelPromise, hfModelPromise])
            .then(([customResult, hfResult]) => {
                lastCustomResult = customResult;
                displayDashboard(customResult, hfResult);
                uploadPage.fadeOut(300, () => resultPage.removeClass('d-none').hide().fadeIn(300));
            })
            .catch((err) => {
                const errorMsg = err.responseJSON ? err.responseJSON.detail : '서버 통신 오류 발생';
                alert(`분석 실패: ${errorMsg}`);
                resetUI();
            })
            .finally(() => {
                loadingOverlay.addClass('d-none');
            });
    }

    function displayDashboard(custom, hf) {
        // 종합 판별 로직 (둘 중 하나라도 AI라고 하면 AI로 의심, 단 신뢰도 고려)
        const customConf = parseFloat(custom.confidence);
        const hfConf = parseFloat(hf.confidence);
        const isCustomFake = custom.prediction === "AI 생성 이미지";
        const isHfFake = hf.prediction === "AI 생성 이미지";

        let finalVerdict = "실제 사진";
        let finalColor = "#22c55e"; // Green
        let finalBg = "#dcfce7";
        let finalConf = Math.max(customConf, hfConf);

        // 간단한 앙상블 로직: 둘 다 Fake면 Fake, 하나만 Fake면 신뢰도 높은 쪽 따름
        if (isCustomFake && isHfFake) {
            finalVerdict = "AI 생성 이미지 (매우 유력)";
            finalColor = "#ef4444"; // Red
            finalBg = "#fee2e2";
        } else if (isCustomFake || isHfFake) {
             if ((isCustomFake && customConf > 50) || (isHfFake && hfConf > 50)) {
                finalVerdict = "AI 조작 의심";
                finalColor = "#f59e0b"; // Orange
                finalBg = "#fef3c7";
             }
        }

        resultVerdictText.text(`${finalVerdict}`);
        resultVerdictText.css('color', finalColor);
        verdictBox.css('background-color', finalBg);

        // 막대그래프 (가로형)
        const barCtx = document.getElementById('probability-bar-chart').getContext('2d');
        if (barChart) barChart.destroy();
        barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Custom 모델 (ELA+LBP)', 'ViT 모델 (Deep Learning)'],
                datasets: [
                    {
                        label: '실제 사진 확률',
                        data: [parseFloat(custom.details.real_prob), parseFloat(hf.details.real_prob)],
                        backgroundColor: '#22c55e',
                        borderRadius: 6,
                        barPercentage: 0.6
                    },
                    {
                        label: 'AI 생성 확률',
                        data: [parseFloat(custom.details.fake_prob), parseFloat(hf.details.fake_prob)],
                        backgroundColor: '#ef4444',
                        borderRadius: 6,
                        barPercentage: 0.6
                    }
                ]
            },
            options: {
                indexAxis: 'y', // 가로 막대
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { beginAtZero: true, max: 100, stacked: true, grid: { display: false } },
                    y: { stacked: true, grid: { display: false }, ticks: { font: { family: 'Pretendard', size: 14 } } }
                },
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true } },
                    tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw.toFixed(1)}%` } }
                }
            }
        });

        // 리포트 채우기
        populateInsights(customInsights, custom.insights);
        populateInsights(hfInsights, hf.insights);

        // 하이라이트 그리기 (이미지 로드 후)
        if (imagePreview[0].complete) {
            drawHighlight();
        } else {
            imagePreview.one('load', drawHighlight);
        }
    }

    function populateInsights(element, insights) {
        element.empty();
        insights.forEach(text => {
            const formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-dark">$1</strong>');
            element.append(`<li class="mb-2"><i class="bi bi-check2-circle me-2 text-primary"></i>${formatted}</li>`);
        });
    }

    function drawHighlight() {
        if (!ctx || !lastCustomResult || !imagePreview.is(':visible')) return;

        highlightCanvas.width = imagePreview.width();
        highlightCanvas.height = imagePreview.height();
        ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

        const area = lastCustomResult.suspicious_area;
        // AI라고 판별된 경우에만 붉은 박스 표시
        if (area && lastCustomResult.prediction === "AI 생성 이미지") {
            const gridSize = area.grid_size;
            const x = (area.hotspot_col / gridSize) * highlightCanvas.width;
            const y = (area.hotspot_row / gridSize) * highlightCanvas.height;
            const w = highlightCanvas.width / gridSize;
            const h = highlightCanvas.height / gridSize;

            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 4;
            ctx.setLineDash([6]); // 점선 효과
            ctx.strokeRect(x, y, w, h);

            // 반투명 붉은 채우기
            ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
            ctx.fillRect(x, y, w, h);
        }
    }

    function resetUI() {
        lastCustomResult = null;
        imageUpload.val('');
        imagePreview.attr('src', '#');
        if (ctx) ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

        resultPage.fadeOut(200, () => uploadPage.fadeIn(200));
        if (barChart) barChart.destroy();
    }

    let resizeTimer;
    $(window).on('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(drawHighlight, 200);
    });
});
