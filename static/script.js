let selectedColor = '#696969FF';
let rectangles = [];
let canvasWidth, canvasHeight;
let isDrawing = false;
let startX, startY;
let ctx;
let uploadedImg, aiImage;


$('.color').click(function () {
    selectedColor = $(this).attr('color');
    $('.color').removeClass('highlight');
    $(this).addClass('highlight');
});

$('#clear').click(function () {
    uploadedImg = undefined;
    rectangles = [];
    clearCanvas();
});

$('#undo').click(function () {
    undo();
});

function clearCanvas() {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        ctx.fillStyle = selectedColor;
}

$('#download-sketch').click(function () {
    downloadCanvasAsPngResized(document.querySelector('#drawing-canvas'));
});

$('#download-final').click(function () {
    downloadCanvasAsPngResized(document.querySelector('#final-canvas'));
});

$('#upload').click(function () {
    var input = document.createElement('input');
    input.setAttribute('type', 'file');
    input.setAttribute('accept', 'image/png,image/jpeg');

    input.addEventListener('change', function (e) {
        const file = e.target.files[0]; // Get the file
        if (!file) return;

        const img = new Image(); // Create a new Image object
        img.onload = function () {
            rectangles = [];
            clearCanvas();
            uploadedImg = img;
            ctx.drawImage(img, 0, 0, 384, 384); // Draw the image
        };
        img.src = URL.createObjectURL(file); // Set the source of the image to the selected file
    });

    input.click();
});


$(function () {
    let canvas = document.querySelector('#drawing-canvas');
    canvasWidth = canvas.width;
    canvasHeight = canvas.height;
    ctx = canvas.getContext('2d');
});

$(document).keydown(function (e) {
    if (e.which === 90 && e.ctrlKey) {
        undo();
    }
});


$('#drawing-canvas')
    .mousedown(function (e) {
        isDrawing = true;
        startX = Math.floor(e.offsetX);
        startY = Math.floor(e.offsetY);
    }).mousemove(function (e) {
    if (isDrawing) {
        drawRect(e);
    }
}).mouseup(function (e) {
    if (isDrawing) {
        [x, y, w, h] = drawRect(e);
        if (w > 0 && h > 0) {
            rectangles.push([x, y, w, h, selectedColor]);
        }
        isDrawing = false;
    }
});


function drawRect(e) {

    clearCanvas();

    if (uploadedImg) {
        ctx.drawImage(uploadedImg, 0, 0, 384, 384);
    }

    rectangles.forEach(([x, y, w, h, c]) => {
        ctx.fillStyle = c;
        ctx.fillRect(x, y, w, h);
    });

    ctx.fillStyle = selectedColor;

    let endX = Math.floor(e.offsetX);
    let endY = Math.floor(e.offsetY);
    let x = startX > endX ? endX : startX;
    let y = startY > endY ? endY : startY;
    console.log(startX, startY, endX, endY, x, y);
    let w = Math.abs(startX - endX);
    let h = Math.abs(startY - endY);
    ctx.fillRect(x, y, w, h);
    return [x, y, w, h];
}

function undo() {
    if (rectangles.length === 0) {
        uploadedImg = undefined;
    } else {
        rectangles.pop();
    }
    clearCanvas();
    if (uploadedImg) {
        ctx.drawImage(uploadedImg, 0, 0, 384, 384);
    }
    rectangles.forEach(([x, y, w, h, c]) => {
        ctx.fillStyle = c;
        ctx.fillRect(x, y, w, h);
    });
    ctx.fillStyle = selectedColor;
}


function downloadCanvasAsPng(canvas) {
    // Create an "invisible" link element
    var downloadLink = document.createElement('a');
    downloadLink.setAttribute('download', 'image.png');

    // Convert canvas content to base64 data URL
    var dataURL = canvas.toDataURL('image/png');

    // Set href attribute of the link to the data URL
    downloadLink.setAttribute('href', dataURL);

    // Click the link programmatically to trigger the download
    downloadLink.click();
}


function getCanvasAsDataUrl(canvas) {
    // Create a temporary canvas element
    var tempCanvas = document.createElement('canvas');
    var tempCtx = tempCanvas.getContext('2d');

    // Set the dimensions of the temporary canvas to 100x100
    tempCanvas.width = 256;
    tempCanvas.height = 256;

    // Draw the canvas content onto the temporary canvas and resize it
    tempCtx.drawImage(canvas, 0, 0, 256, 256);

    return tempCanvas.toDataURL('image/jpeg');
}

function downloadCanvasAsPngResized(canvas) {

    // Create an "invisible" link element
    var downloadLink = document.createElement('a');
    downloadLink.setAttribute('download', 'image.png');

    // Set href attribute of the link to the resized data URL
    downloadLink.setAttribute('href', getCanvasAsDataUrl(canvas));

    // Click the link programmatically to trigger the download
    downloadLink.click();
}


function sendToServer() {
    var imageDataURL = getCanvasAsDataUrl(
        document.querySelector('#drawing-canvas')
    );
    $.ajax({
        type: "POST",
        url: "/process",
        data: {
            image_data: imageDataURL
        },
        success: function (response) {
            $('.loader-wrapper').removeClass('show').addClass('hide');
            $('#final-canvas').addClass('show').removeClass('hide');
            var fCanvas = document.getElementById('final-canvas'); // Get the second canvas
            var ctx = fCanvas.getContext('2d');
            aiImage = new Image();
            aiImage.onload = function () {
                ctx.drawImage(aiImage, 0, 0, 384, 384); // Draw the image on the canvas
            };
            aiImage.src = response.image;
        },
        error: function (error) {
            console.error("Error sending canvas data to server:", error);
        }
    });
    $(this).remove();
    $('.loader-wrapper').removeClass('hide').addClass('show');
}


$('.lucide-sparkles').click(sendToServer);


$('.generate').click(function () {
    $('.generate').addClass('hide').removeClass('show');
    sendToServer();
});

$('#run').click(function () {
    sendToServer();
    $('.loader-wrapper').addClass('show').removeClass('hide');
    $('#final-canvas').removeClass('show').addClass('hide');
});

$('#move-to-editor').click(function () {
    if(aiImage) {
        uploadedImg = aiImage;
        clearCanvas();
        rectangles = [];
        ctx.drawImage(uploadedImg, 0, 0, 384, 384);
    }
});

