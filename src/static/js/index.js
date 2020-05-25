window.jquery = require("jquery");
window.Jquery = require("jquery");
window.$ = require("jquery");
require("../css/bootstrap/bootstrap.min.css");
require("../css/fonts/fontawesome.min.css");
require("../css/client.css");
require("../css/percentage_circle.css");
require('bootstrap')
require('../js/fontawesome/all.min')
window.Handlebars = require("./handlebars/handlebars.min");
require("./handlebars/handlebars.runtime.min");
require("./view");

const processController = require("./controller/processController");

$(document).ready(function() {
    window.processManager = new processController()
    processManager.startProcess()
});