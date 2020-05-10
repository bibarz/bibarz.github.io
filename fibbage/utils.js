var setCookie = function(cname, cvalue, exp_secs) {
  var d = new Date();
  d.setTime(d.getTime() + exp_secs*1000);
  var expires = "expires="+ d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
  var name = cname + "=";
  var decodedCookie = decodeURIComponent(document.cookie);
  var ca = decodedCookie.split(';');
  for(var i = 0; i <ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

;(function($) {
    $.fn.textfill = function(options) {
        var ourText = $(this);
        var fontSize = parseInt(ourText.css('font-size'));
        var ourDiv = ourText.parent();
        var maxHeight = ourDiv.height();
        var maxWidth = (7*ourDiv.width())/8;
        console.log("ourDiv:", ourDiv, "ourText:", ourText, "fontsize:", fontSize, "maxWidth", maxWidth)
        var textHeight;
        var textWidth;
        do {
            ourText.css('fontSize', fontSize);
            textHeight = ourText.height();
            textWidth = ourText.width();
            fontSize = fontSize - 1;
        } while ((textHeight > maxHeight || textWidth > maxWidth) && fontSize > 3);
        return this;
    }
})(jQuery);
