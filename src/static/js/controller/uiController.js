const historyController = require("./historyController")
const scannerController = require("./scannerController")

module.exports = class uiController {
    constructor() {
        window.historyManager = new historyController()
    }
    prepare() {
        return new Promise((resolve, reject) => {
            this.prepareBaseUI();
            this.prepareContentUI();
            this.prepareEventHandler();
            return resolve(true);
        })
    }
//    original
//    prepareBaseUI() {
//        var headerTemplate = Handlebars.templates.header()
//        var historyTemplate = Handlebars.templates.historyPanel()
//        var scannerTemplate = Handlebars.templates.scanner()
//        var structure = {
//            "header": headerTemplate,
//            "scanner": scannerTemplate,
//            "history": historyTemplate,
//        }
//        var webStructure = Handlebars.templates.structure(structure)
//        $('body').html(webStructure)
//    }
    prepareBaseUI() {
        var headerTemplate = Handlebars.templates.header()
        var scannerTemplate = Handlebars.templates.scanner()
        var predictionTemplate = Handlebars.templates.predictionPanel()
        var structure = {
            "header": headerTemplate,
            "scanner": scannerTemplate,
            "history": predictionTemplate,
        }
        var webStructure = Handlebars.templates.structure(structure)
//        window.setInterval(this.refresh, 1000, webStructure)
        $('body').html(webStructure)
    }
    refresh(webStructure) {
        console.log("refreshed")
        $('body').html("")
        $('body').html(webStructure)
    }

    prepareContentUI () {
//        window.historyManager.getHistoryTemplate()
//        .then(result =>{
//            $('#history-cards-container').append(result);
//        })
    }
    prepareEventHandler() {
        window.scanner = new scannerController()

        $('#btn-header-start').click(function(){
            $('#btn-header-start').hide();
            $('#btn-header-stop').show();
//            $('#btn-scanner-capture').attr('disabled', false).show()
            window.scanner.startScanner();
        })

        $('#btn-header-stop').click(function(){
            $('#btn-header-stop').hide();
            $('#btn-header-start').show();
//            $('#btn-scanner-capture').attr('disabled', true).hide()
            if (window.stream) {
                window.stream.getTracks().forEach(function(track) {
                    track.stop();
                });
            }
            window.scanner.stopScanner();
            $('#prediction-panel').html("");
            $('#bounding-box').hide();
        })

//        $('#btn-scanner-capture').click(function(){
////            window.scanner.capture()
//            window.scanner.stream()
//        })
    }
    showResult(result) {
        var self = this;
        if (result['isItem']){
            var prediction_html = "";
            result['prob'].forEach(function(item, index){
                var predicted_result = {
                    "product_name": result['name'][index],
                    "probability": parseInt(parseFloat(item) * 100).toString(),
                    "category": "None",
                    "product_id": result['class'][index].toString()
                };
                var predictionTemplate = Handlebars.templates.predictionCard(predicted_result);
                prediction_html += predictionTemplate
            })
//            var predicted_result = {
//                "product_name": "ABS V/Socket BBB 40mm",
//                "probability": parseInt(parseFloat(result['sorted_prob'][0]) * 100).toString(),
//                "category": "Iron",
//                "product_id": "1004"
//            };
//            var predictionTemplate = Handlebars.templates.predictionCard(predicted_result);
            $('#prediction-panel').html("")
            $('#prediction-panel').html(prediction_html)
            self.showBoundingBox(result);
        }
        else {
            $('#prediction-panel').html("")
            $("#bounding-box").hide();
        }
    }
    showBoundingBox(result) {
        var scanner_height = $("#video-scanner").height();
        var scanner_width = $("#video-scanner").width();
        console.log(scanner_height, scanner_width)
        var top = parseInt(parseFloat(result['top']) * scanner_height);
        var bottom = parseInt(parseFloat(result['bottom']) * scanner_height);
        var left = parseInt(parseFloat(result['left']) * scanner_width);
        var right = parseInt(parseFloat(result['right']) * scanner_width);
        console.log(top.toString()+"px", left.toString()+"px", (right-left).toString()+"px", (bottom-top).toString()+"px");
//        $("#bounding-box").css({"top": "10px", "left": "50px"});
        $("#bounding-box").css("top", top.toString()+"px");
        $("#bounding-box").css("left", left.toString()+"px");
        $("#bounding-box").css("width", (right-left).toString()+"px");
        $("#bounding-box").css("height", (bottom-top).toString()+"px");
        $("#bounding-box").show();
    }
//    constructCard() {
//        return new Promise((resolve, reject) => {
//            var current_time = Date.now().toString()
//            var cardTemplate = Handlebars.templates.historyCard({
//                "canvas_id" : current_time,
//                "card_id": current_time + '-card',
//                "progress_id": current_time + '-progress',
//                "item_id": current_time + '-item',
//            })
//            $('#history-cards-container').prepend(cardTemplate)
//            console.log('inserted empty card')
//            return resolve(current_time)
//        })
//    }
//    showCard(result, timestamp) {
//        var card_id = '#'+ timestamp + '-card';
//        var progress_id = '#' + timestamp + '-progress';
//        var item_id = '#' + timestamp + '-item';
//        var resultData = JSON.parse(result);
//        var confidence_level = resultData['confidence_level']+'%';
//        $(item_id).html('Item ' + resultData['predicted_class'])
//        $(progress_id).css('width', confidence_level);
//        $(progress_id).children('span').html('confidence_level: ' + confidence_level);
//        $(card_id).show();
//        this.disloading();
//    }
    loading() {
        $('#loading-bar').show()
        $('#btn-scanner-capture').attr('disabled', true)
    }
    disloading() {
        $('#loading-bar').hide()
        $('#btn-scanner-capture').attr('disabled', false)
    }
}