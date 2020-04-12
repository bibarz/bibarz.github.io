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
    $.fn.textfillparent = function(options) {
        var fontSize = $(window).width() / 6;
        var ourText = $(this);
        var maxHeight = $(this).parent().height();
        var maxWidth = $(this).parent().width();
		var textHeight = ourText.outerHeight();
		var textWidth = ourText.outerWidth();
		// console.log("maxHeight " + maxHeight + " maxWidth " + maxWidth + " textHeight " + textHeight + " textWidth " + textWidth);
		if (textHeight < maxHeight && textWidth < maxWidth) return this;
        do {
            ourText.css('fontSize', fontSize);
            textHeight = ourText.outerHeight();
            textWidth = ourText.outerWidth();
			// console.log("fontSize " + fontSize + " textHeight " + textHeight + " textWidth " + textWidth);

            fontSize = fontSize - 1;
        } while ((textHeight > maxHeight || textWidth > maxWidth) && fontSize > 3);
        return this;
    }
})(jQuery);
