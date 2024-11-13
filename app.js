async function uploadFile(fileType) {
    const fileInput = document.getElementById("csv-file-" + fileType);

    const formData = new FormData();
    if (fileInput.files.length > 0) {  
        formData.append("file", fileInput.files[0]);
    } else {
        console.error("No file selected");
        return; 
    }

    try {
        const response = await fetch('/process_csv', {
            method: 'POST',
            body: formData
        });

       
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log(data);  
    } catch (error) {
        console.error('Error:', error);  
    }
}
