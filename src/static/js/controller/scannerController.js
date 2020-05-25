module.exports = class scannerController{
    constructor() {
        console.log("scannerController.....")
        var self = this;
        this.constraints = {
              audio: false,
              video: true
         };
        navigator.mediaDevices.enumerateDevices()
        .then(function(devices) {
            devices.forEach(function(device) {
                if (device.label.split(" ")[0] == "Webcam"){
                    self.constraints = {
                      audio: false,
                      video: { deviceId: device.deviceId }
                    };
                }
            });
            console.log(self.constraints)
        })
        .catch(function(err) {
            console.log(err)
        })

        this.video = document.querySelector('video');
        this.stream_process = null;
    }

    startScanner() {
        var self = this;
        navigator.mediaDevices.getUserMedia(this.constraints).
        then(stream => {
            window.stream = stream; // make stream available to browser console
            self.video.srcObject = stream;
            self.stream();
        })
        .catch(err => {
            console.log('navigator.MediaDevices.getUserMedia error: ', err.message, err.name);
        });
    }

    stopScanner() {
        console.log('stopping scanner')
        clearInterval(this.stream_process);
        this.stream_process = null;
    }

    capture(self) {
        var canvas = $('#video-canvas')[0];
        var ctx = canvas.getContext('2d')
        canvas.width = self.video.videoWidth/3;
        canvas.height = self.video.videoHeight/3;
        console.log("canvas size:", canvas.width, canvas.height)
        ctx.drawImage(self.video, 0, 0, canvas.width, canvas.height);
//        this.download_image(canvas)
//        .then(function(){
//            console.log("downloaded")
//        });
        self.get_prediction_result(canvas)
        .then(result =>{
            result = JSON.parse(result);
            console.log('result',result);
            uiManager.showResult(result);
        })
    }

//    capture() {
//        console.log("capturing.....")
//        var self = this;
//        window.uiManager.loading();
//        window.uiManager.constructCard()
//        .then(timestamp => {
//            var canvas_id = '#'+ timestamp;
//            var canvas = $(canvas_id)[0];
//            var ctx = canvas.getContext('2d')
//            canvas.width = self.video.videoWidth/3;
//            canvas.height = self.video.videoHeight/3;
//            console.log("canvas size:", canvas.width, canvas.height)
//            ctx.drawImage(self.video, 0, 0, canvas.width, canvas.height);
////            var imageData = ctx.getImageData(0, 0, canvas.width/3, canvas.height/3);
////            console.log('Image Data', imageData);
//            var dataurl_download = canvas.toDataURL('image/jpeg', 1.0);
//            var dataurl = canvas.toDataURL();
//            console.log("data url", dataurl)
//            self.get_prediction_result(dataurl)
//            .then(result =>{
//            console.log('result',result)
//                var result_json = result;
//                if (result_json['isItem']) {
//                    console.log('Item found.')
//                    window.uiManager.showCard(result, timestamp);
//                }
//                else {
//                    console.log('No item found.')
//                }
//            })
//            self.download_image(dataurl_download)
//            .then(function(){
//                console.log("downloaded")
////                setTimeout(function(){
////                    console.log('get predict')
////                    self.get_prediction_result()
////                    .then(result => {
////                        console.log('show card')
////                        window.uiManager.showCard(result, timestamp);
////                    });
////                },2000)
//            });
//        });
//    }

    stream() {
        this.stream_process = setInterval(this.capture, 2000, this)
    }

    get_prediction_result(canvas) {
        var img_data = canvas.toDataURL();
//        console.log("img_data", img_data)
        return new Promise((resolve, reject) => {
            $.ajax({
            type: "POST",
            url: "http://localhost:5000/api/predict",
            data: {
                img: img_data
            },
            success: function(resultData) {
                return resolve(resultData);
            },
            error: function(request, status, errorThrown) {
                return reject(errorThrown)
            },
            }).done(function(o) {
                console.log('saved');
                // If you want the file to be visible in the browser
                // - please modify the callback in javascript. All you
                // need is to return the url to the file, you just saved
                // and than put the image in your browser.
            });
        });
    }
    download_image(canvas) {
        var url =  canvas.toDataURL('image/jpeg', 1.0);
        return new Promise((resolve, reject) => {
            var a = document.createElement("a");
            a.href = url;
            a.setAttribute("download", 'test.jpeg');
            a.click()
            return resolve(true);
        })
    }
}