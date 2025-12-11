$(document).ready(function() {
    // HTML 요소 변수
    const uploadPage = $('#upload-page');
    const resultPage = $('#result-page');
    const loadingOverlay = $('#loading-overlay');

    const dropZone = $('#drop-zone');
    const imageUpload = $('#image-upload');
    const imageDisplayArea = $('#image-display-area');
    const imagePreview = $('#image-preview');
    const highlightCanvas = $('#highlight-canvas')[0];
    const ctx = highlightCanvas ? highlightCanvas.getContext('2d') : null;

    const customInsights = $('#custom-insights');
    const hfInsights = $('#hf-insights');
    const resetButton = $('#reset-button');
    // const resultVerdictText = $('#result-verdict-text'); // 비교 UI에서는 사용 안 함

    let lineChart, barChart;
    let lastCustomResult = null;
    let selectedFile = null;

    // --- 이벤트 핸들러 ---
    dropZone.on('click', () => imageUpload.click());
    dropZone.on('dragover', (e) => { e.preventDefault(); dropZone.addClass('dragover'); });
    dropZone.on('dragleave', (e) => { e.preventDefault(); dropZone.removeClass('dragover'); });
    dropZone.on('drop', (e) => {
        e.preventDefault();
        dropZone.removeClass('dragover');
        if (e.originalEvent.dataTransfer.files.length) {
            handleFile(e.originalEvent.dataTransfer.files[0]);
        }
    });
    imageUpload.on('change', function() {
        if (this.files.length) {
            handleFile(this.files[0]);
        }
    });
    resetButton.on('click', resetUI);

    // --- 함수 로직 ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 선택할 수 있습니다.'); return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.attr('src', e.target.result);
            runBothAnalyses(file); // ★★★ 파일 읽기가 끝나면 바로 분석 실행 ★★★
        }
        reader.readAsDataURL(file);
    }

    function runBothAnalyses(file) {
        if (!file) {
             console.error("분석할 파일이 없습니다.");
             return;
        }

        loadingOverlay.removeClass('d-none');
        uploadPage.fadeOut(200);

        const formData = new FormData();
        formData.append('file', file);

        // ★★★ 두 개의 API를 동시에 호출 ★★★
        const customModelPromise = $.ajax({
            url: 'http://127.0.0.1:8000/analyze-with-custom-model/',
            type: 'POST', data: formData, processData: false, contentType: false
        });

        const hfModelPromise = $.ajax({
            url: 'http://127.0.0.1:8000/analyze-with-huggingface-model/',
            type: 'POST', data: formData, processData: false, contentType: false
        });

        // ★★★ 두 API의 응답을 모두 받은 후에 결과 표시 ★★★
        Promise.all([customModelPromise, hfModelPromise])
            .then(([customResult, hfResult]) => {
                lastCustomResult = customResult; // 하이라이트 그리기를 위해 저장
                displayDashboard(customResult, hfResult); // 두 결과를 모두 전달
                resultPage.removeClass('d-none').hide().fadeIn(500);
            })
            .catch((err) => {
                const errorMsg = err.responseJSON ? err.responseJSON.detail : '하나 이상의 모델 분석에 실패했습니다.';
                alert(`분석 실패: ${errorMsg}`);
                resetUI();
            })
            .finally(() => {
                loadingOverlay.addClass('d-none');
            });
    }

    function displayDashboard(custom, hf) {
        const customConf = parseFloat(custom.confidence);
        const hfConf = parseFloat(hf.confidence);

        // 꺾은선 그래프 (신뢰도 비교)
        const lineCtx = document.getElementById('confidence-line-chart')?.getContext('2d');
        if(lineCtx){
            if (lineChart) lineChart.destroy();
            lineChart = new Chart(lineCtx, {
                type: 'line',
                data: {
                    labels: ['직접 만든 모델 (ELA+LBP)', '허깅페이스 모델 (ViT)'],
                    datasets: [{
                        label: '최종 판별 신뢰도 (%)',
                        data: [customConf, hfConf],
                        borderColor: '#0d6efd', backgroundColor: 'rgba(13, 110, 253, 0.2)',
                        fill: true, tension: 0.1, pointRadius: 5, pointBackgroundColor: '#0d6efd'
                    }]
                },
                options: chartOptions(true)
            });
        }

        // 그룹 막대그래프 (상세 확률 비교)
        const barCtx = document.getElementById('probability-bar-chart')?.getContext('2d');
        if(barCtx){
            if (barChart) barChart.destroy();
            barChart = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: ['직접 만든 모델 (ELA+LBP)', '허깅페이스 모델 (ViT)'],
                    datasets: [
                        { label: '실제 사진 확률', data: [parseFloat(custom.details.real_prob), parseFloat(hf.details.real_prob)], backgroundColor: '#5cb85c' },
                        { label: 'AI 생성 확률', data: [parseFloat(custom.details.fake_prob), parseFloat(hf.details.fake_prob)], backgroundColor: '#d9534f' }
                    ]
                },
                options: chartOptions(false)
            });
        }

        // 분석 리포트 채우기
        populateInsights(customInsights, custom.insights);
        populateInsights(hfInsights, hf.insights);

        // 이미지 로드 후 하이라이트 그리기
        imagePreview.off('load').on('load', drawHighlight);
        if (imagePreview[0].complete) imagePreview.trigger('load');
    }

    function populateInsights(element, insights) {
        if (!element || !insights) return;
        element.empty();
        insights.forEach(text => {
            const formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            element.append(`<li>- ${formatted}</li>`);
        });
    }

    function drawHighlight() {
        if (!ctx || !lastCustomResult || !imagePreview.is(':visible')) return;
        highlightCanvas.width = imagePreview.width();
        highlightCanvas.height = imagePreview.height();
        ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
        const area = lastCustomResult.suspicious_area;
        if (area) {
            const gridSize = area.grid_size;
            const x = (area.hotspot_col / gridSize) * highlightCanvas.width;
            const y = (area.hotspot_row / gridSize) * highlightCanvas.height;
            const w = highlightCanvas.width / gridSize;
            const h = highlightCanvas.height / gridSize;
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);
        }
    }

    function chartOptions(isLine) {
        const textColor = '#6c757d';
        const primaryTextColor = '#212529';
        return {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { color: textColor, callback: (v) => v + '%' }, grid: { color: '#dee2e6' } },
                x: { ticks: { color: primaryTextColor, font: { size: 14 } }, grid: { display: false } }
            },
            plugins: {
                legend: { display: !isLine, labels: { color: primaryTextColor } },
                tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.raw.toFixed(2)}%` } }
            }
        };
    }

    function resetUI() {
        lastCustomResult = null;
        selectedFile = null;
        imageUpload.val('');
        imagePreview.attr('src', '#');
        if (ctx) ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

        resultPage.fadeOut(200, () => uploadPage.fadeIn(200));
        if (lineChart) lineChart.destroy();
        if (barChart) barChart.destroy();
    }

    let resizeTimer;
    $(window).on('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(drawHighlight, 250);
    });
});