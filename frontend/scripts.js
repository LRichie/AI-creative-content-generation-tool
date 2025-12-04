document.addEventListener('DOMContentLoaded', () => {
    const invokeAgentButton = document.getElementById('invoke-agent');
    const resumeAgentButton = document.getElementById('resume-agent');
    const generatedTextTextarea = document.getElementById('generated-text');
    const infoTextarea = document.getElementById('info');
    const sessionIdInput = document.getElementById('session-id');

    // Generate a UUID for the session ID when the page loads
    const sessionId = uuidv4();
    sessionIdInput.value = sessionId;

    invokeAgentButton.addEventListener('click', async () => {
        const style = document.getElementById('style').value;
        const space = document.getElementById('space').value;
        const emotion = document.getElementById('emotion').value;
        const platform = document.getElementById('platform').value;
        const imageInput = document.getElementById('image').files[0];

        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('style', style);
        formData.append('space', space);
        formData.append('emotion', emotion);
        formData.append('platform', platform);
        if (imageInput) {
            formData.append('image', imageInput);
        }

        try {
            const response = await fetch('/api/agent/invoke', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            generatedTextTextarea.value = result.review_content;
        } catch (error) {
            console.error('Error invoking agent:', error);
            alert('调用代理失败，请重试。');
        }
    });

    resumeAgentButton.addEventListener('click', async () => {
        // const decision = document.getElementById('decision').value;
        const info = infoTextarea.value;

        const requestBody = {
            session_id: sessionId,
            // decision: decision,
            // info: decision === 'edit' ? info : null
            info: info
        };

        try {
            const response = await fetch('/api/agent/resume', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            generatedTextTextarea.value = result.review_content;

            if (result.status === 'completed') {
                alert('最终内容生成成功！');
            }
        } catch (error) {
            console.error('Error resuming agent:', error);
            alert('恢复代理失败，请重试。');
        }
    });

    function uuidv4() {
        return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }
});
