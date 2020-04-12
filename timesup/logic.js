var debug = true;
// var is_chief = (navigator.userAgent.indexOf("Chrome") == -1) && (navigator.userAgent.indexOf("Firefox") == -1) && (navigator.userAgent.indexOf("iPhone") == -1);
var is_chief = window.location.search.substring(1).indexOf("chief") >= 0;
// the_gs is mainly for chief, it is where it keeps the game state.
// Non-chiefs use it only for two reasons:
//    - To detect the first message from timesup/gamestate: so we always display
//		(otherwise, if we reload the page and the last gamestate is non-refreshing
//		we would see nothing). the_gs is null on reload, that's how we detect the
//		first message.
//	  - For certain callbacks that need the current gamestate, even if there was
//		no new display to update the callback itself. E.g., the proposing and voting
//		callbacks that need to know how many people have proposed or voted so far,
//		but won't see the update because they are non-refreshing.
var the_gs = null;
var all_cards = [];
var beep = new Audio('short_countdown.mp3');
var alarm_sound = new Audio('alarm.mp3');

// Work around mobile browsers blocking sounds unless responding
// to user input: play each sound once on the first user interaction
// (plzy and pause immediately so nothing is heard)
var allAudio = [];
allAudio.push(beep);
allAudio.push(alarm_sound);
var tapped = function() {
	if(allAudio) {
		for(var audio of allAudio) {
			audio.play();
			audio.pause();
			audio.currentTime = 0;
		}
		allAudio = null
	}
}
document.body.addEventListener('touchstart', tapped, false);
document.body.addEventListener('click', tapped, false);

var diffusion_params = {
		host   : 'klander-l102aw.eu.diffusion.cloud',
		port   : 443,
		secure : true,
		principal : 'admin',
		credentials : 'admin'
	}
var TopicSpecification = diffusion.topics.TopicSpecification;
var TopicType = diffusion.topics.TopicType;


var add_cards_topic_callback = function(path, newValue) {
	if (debug) console.log('Got string update for topic: ' + path, newValue);
	var cards = JSON.parse(newValue.get());
	for (var i=0; i < cards.length; i++) {
		var card = cards[i].trim();
		if (all_cards.indexOf(card) == -1) all_cards.push(card);
	}
	if (debug) console.log("Cards are now " + all_cards);
	$("#manage_container p").text("Tarjetas: " + all_cards.length);
}


var send_cards = function(session) {
	// Note: this doesn't work:
	//	cards = $("#manage_container li").map(() => $(this).text()).get();
	// because arrow functions take "this" from the surrounding lexical context,
	// while properly declared functions define "this" from their caller.
	cards = $("#manage_container li").map(function() {return $(this).text();}).get();
	if (debug) console.log("Sending cards " + cards);
	session.topics.updateValue('timesup/cards',
							   JSON.stringify(cards),
							   diffusion.datatypes.json());
	$("#manage_container li").remove();
}


var start_button_click_callback = function(session) {
	if (all_cards.length <2) {
		alert("Cannot play with less than 2 cards.");
		return;
	}
	the_gs = new GameState(all_cards);
	chief_start_game(session);
	execute_command(session, the_gs, {"name": "init"});
	setCookie("timesup_game_is_on", true, 3600 * 3);  // game on for 3 hours
}


var setup_cards = function(session) {
	var player_id = getCookie("player_id");
	$("div#game_container").hide();
	$("div#manage_container").show();
	if (!player_id) {
		player_id = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
		if (debug) console.log("Registering new player id " + player_id);
		setCookie("player_id", player_id);
	}
	var send_elem = function() {
		var todoText = $("input[type='text']").val();
		if (!todoText) return;
		$("input[type='text']").val("");
		//create a new li and add to ul
		$("ul").append("<li><span><i class='fa fa-trash'></i></span> " + todoText + "</li>");
	}
	$("#manage_container form").on("submit", function(event){
		event.preventDefault();
		send_elem();
	});
	$("#manage_container input[type='text']").on("focusout", function(event){
		send_elem();
	});
	//Click on X to delete Todo
	$("ul").on("click", "span", function(event){
		$(this).parent().fadeOut(500,function(){
			$(this).remove();
		});
		event.stopPropagation();
	});
	$("#send_cards").click(() => {send_cards(session);});
	if (!is_chief) {
		$("#start_game").hide();
	} else {
		$("#start_game").show();
	}

	if (!is_chief) subscribe_to_gamestate(session);
}


var setup_game = function(session) {
	return session.topics.remove('timesup/gamestate')
	.then(() => {
		console.log('Removed topic timesup/gamestate');
		}, reason => {
		console.log('Failed to remove topic timesup/gamestate: ', reason);
	})
	.then(() => {
		return session.topics.remove('timesup/cards')})
	.then(() => {
		return session.topics.add('timesup/cards',
								  new TopicSpecification(TopicType.JSON,
														 {DONT_RETAIN_VALUE: "true"}))})
	.then(result => {
		console.log('Added topic: ' + result.topic);
		session
			.addStream('timesup/cards', diffusion.datatypes.json())
			.on('value', function(path, specification, newValue, oldValue) {
				add_cards_topic_callback(path, newValue);
			});
		// Subscribe to the topic. The value stream will now start to receive notifications.
		session.select('timesup/cards');
		}, reason => {
			console.log('Failed to add topic: ', reason);
		})
	.then(() => {
		$("#start_game").click(() => {start_button_click_callback(session);});
		return session;
	});
}

var subscribe_to_gamestate = function(session) {
	session
		.addStream('timesup/gamestate', diffusion.datatypes.json())
		.on('value', function(path, specification, newValue, oldValue) {
			var is_first = (the_gs === null);
			gs = JSON.parse(newValue.get());
			var refresh = gs.refresh;
			if (debug) console.log('Got JSON update for topic: ' + path, gs);
			if (is_first) {
				the_gs = new GameState(gs.deck);
			}
			the_gs = Object.assign(the_gs, gs);
			if (refresh || is_first) display(session, gs);
			if (is_first && is_chief) {
				if (debug) console.log("Subscribing to timesup/command after receiving timesup/gamestate, the_gs is ", the_gs);
				subscribe_to_command(session, the_gs);
			}
		});
	// Subscribe to the topic. The value stream will now start to receive notifications.
	session.select('timesup/gamestate');
}

var subscribe_to_command = function(session, gs) {
	if (debug) console.log("Subscribing to timesup/command, gs is ", gs);
	session
		.addStream('timesup/command', diffusion.datatypes.json())
		.on('value', function(path, specification, newValue, oldValue) {
			if (debug) console.log('Got JSON update for topic: ' + path, newValue.get());
			execute_command(session, gs, JSON.parse(newValue.get()));
		});
	// Subscribe to the topic. The value stream will now start to receive notifications.
	session.select('timesup/command');
}


var chief_start_game = function(session) {
	session.topics.add('timesup/gamestate',
					   new TopicSpecification(TopicType.JSON)).then(function(result) {
        console.log('Added topic: ' + result.topic);
		subscribe_to_gamestate(session);
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});

    session.topics.add('timesup/command',
					   new TopicSpecification(TopicType.JSON,
											  {DONT_RETAIN_VALUE: "true"})).then(function(result) {
        console.log('Added topic: ' + result.topic);
		// When called from the start button, the chief has the_gs already
		// and can and should start to listen to commands.
		// (The alternative is starting from reload, we will get
		// the gs on the first message after subscribing to timesup/game_state)
		if (the_gs) {
			if (debug) console.log("Subscribing to timesup/command " +
								   "on start game, the_gs is " + the_gs);
			subscribe_to_command(session, the_gs);
		}
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});
}

var pub_command = function(session, obj) {
	if (!session) {
		console.log("No session, couldn't pub command " + obj);		
	} else {
		if (debug) {
			console.log("Publishing command with name " + obj["name"])
		}
		session.topics.updateValue('timesup/command',
								   JSON.stringify(obj),
								   diffusion.datatypes.json());
	}
}

var execute_command = function(session, gs, obj) {
	if (debug) {
		console.log("Executing command " + obj);
		console.log("gs is ", gs);
		console.log("the_gs is ", the_gs);
	}
	var refresh = false;
	if (obj["name"] == "init") {
		refresh = gs.init();
	}
	if (obj["name"] == "new_round") {
		refresh = gs.new_round(obj["player_id"], obj["player_time"]);
	}
	if (obj["name"] == "end_round") {
		refresh = gs.end_round(obj["cards_guess"], obj["cards_pass"], obj["cards_delete"]);
	}
	if (obj["name"] == "round_time") {
		refresh = gs.set_round_time(obj["time"]);
	}
	gs.refresh = refresh;
	session.topics.updateValue('timesup/gamestate',
							   JSON.stringify(gs),
							   diffusion.datatypes.json());
}


var get_stage_text = function(i) {
	if (i==0) return "Primera fase";
	if (i==1) return "Segunda fase";
	if (i==2) return "Tercera fase";
	return "Fase " + (i+1);
}


var display = function(session, gs) {
	player_id = getCookie("player_id");
	$("#manage_container").hide();
	$("#new_game").hide();
	$("#game_container").show();
	var beeped = false;
	if (gs.player_id < 0) {  // no round is on
		if (debug) {
			console.log("No round active.");
			console.log("Deck: " + gs.deck);
			console.log("Deck index: " + gs.deck_index);
		}
		$("#in_round").hide();
		$("#waiting_round").hide();
		$("#before_round").show();
		var remaining = gs.deck.length - gs.deck_index;
		var text_before = (remaining == 1) ? "Queda ":"Quedan ";
		var text_after = (remaining == 1) ? " tarjeta":" tarjetas";
		$("#stage_number p").text(get_stage_text(gs.stage));
		$("#before_round .cards_left_text p").text(
			text_before + remaining + text_after);
		$("#start_round").off("click").on("click", () => {
			pub_command(session, {
				name: "new_round",
				player_id: player_id,
				player_time: new Date().getTime(),
			});
		});
		if (gs.guessed_last_round >= 0) {
			$("#last_round_result p").show().html(
				"Puntos en la &uacute;ltima ronda: " + gs.guessed_last_round);
		} else {
			$("#last_round_result p").hide();
		}
		$("#config #round_time").attr("placeholder", gs.round_time);
		var send_round_time = function() {
			if ($("#config #round_time").val()) {
				pub_command(session, {
					name: "round_time",
					time: $("#config #round_time").val(),
				});
			}
		}
		$("#config form").on("submit", function(event){
			event.preventDefault();
			send_round_time();
		});
		$("#config input[type='text']").on("focusout", function(event){
			send_round_time();
		});
		if (is_chief) {
			$("#new_game").show().off("click").on("click", () => {
				var encoded = "&iquest;Seguro?";
				var decoded = $("<div/>").html(encoded).text();
				result = confirm(decoded);
				if (result) {
					setCookie("timesup_game_is_on", false, -1);
					$("#game_container").hide();
					$("#manage_container").show();
					if (debug) { console.log("Setting up game.");}
					setup_game(session).then(setup_cards);
				}
			});
		}
	} else if (gs.player_id != player_id) {
		$("#in_round").hide();
		$("#waiting_round").show();
		$("#before_round").hide();
		$("#waiting_round p").html("Ronda en juego");
	} else {
		$("#in_round").show();
		$("#waiting_round").hide();
		$("#before_round").hide();
		var deck_index = gs.deck_index;
		var cards_guess = [];
		var cards_pass = [];
		var cards_delete = [];
		var display_card = function() {
			var remaining = gs.deck.length - deck_index;
			$("#candidate_text p").css('fontSize', 'min(24vw,18vh)')
				.text(gs.deck[deck_index].trim())
				.textfillparent();
			var text = (remaining == 1) ? "Queda ":"Quedan ";
			$("#in_round .cards_left_text p").text(text + remaining);
		}
		var end_round = function () {
			alarm_sound.play();
			beeped = false;
			clearInterval(clock_var);
			pub_command(session, {
				name: "end_round",
				cards_guess: cards_guess,
				cards_pass: cards_pass,
				cards_delete: cards_delete,
			});			
			$("#game_container").hide();
		};
		var clock_var = setInterval(() => {
			var seconds = (new Date().getTime() - gs.player_start_time) / 1000;
			var time_left = gs.round_time - seconds;
			const time_warn = 5;
			if (time_left < time_warn) {
				var phase = (1 + Math.cos(2 * 3.1416 * (time_warn - time_left))) / 2;
				var green = Math.round(100 + 76 * phase);
				var red = 112 + Math.round(76 * (1 - phase));
				$("#clock p").css("background-color", "rgb(" + red + "," + green + ",112)");
			}
			if (time_left < 4.3 && !beeped) {
				beeped = true;
				beep.play();
			}
			if (time_left > 0) $("#clock p").text(time_left.toFixed(1));
			else end_round();
		}, 50);
		var next_card = function() {
			deck_index++;
			var remaining = gs.deck.length - deck_index;
			if (remaining <= 0) end_round();
			else display_card();
		}
		$("#correct").off("click").on("click", () => {
			cards_guess.push(gs.deck[deck_index]);
			next_card();
		});
		if (gs.stage > 0) {
			$("#pass").show();
			$("#pass").off("click").on("click", () => {
				cards_pass.push(gs.deck[deck_index]);
				next_card();
			});
			$("#cheat").off("click").hide();
		} else {
			$("#pass").off("click").hide();
			$("#cheat").show();
			$("#cheat").off("click").on("click", () => {
				cards_pass.push(gs.deck[deck_index]);
				next_card();
			});
		}
		$("#delete").off("click").on("click", () => {
			cards_delete.push(gs.deck[deck_index]);
			next_card();
		});
		display_card();
	}
}

var session_promise = diffusion.connect(diffusion_params);
session_promise.catch(error => {alert("Error setting up Diffusion session:" + error)});

if (is_chief) {
	if (!getCookie("timesup_game_is_on")) {
		$("#game_container").hide();
		$("#manage_container").show();
		if (debug) { console.log("Setting up game.");}
		session_promise.then(setup_game).then(setup_cards);
	} else {
		console.log("timesup_game_is_on value " + getCookie("timesup_game_is_on"));		
		if (debug) { console.log("Game is on.");}
		session_promise.then(chief_start_game);
	}
} else {
	$("#manage_container").hide();
	if (debug) { console.log("Connecting to game.");}
	session_promise.then(setup_cards);
}


	