var debug = true;
// var is_chief = (navigator.userAgent.indexOf("Chrome") == -1) && (navigator.userAgent.indexOf("Firefox") == -1);
var is_chief = window.location.search.substring(1).indexOf("chief") >= 0;
// the_gs is mainly for chief, it is where it keeps the game state.
// Non-chiefs use it only for two reasons:
//    - To detect the first message from dixit/gamestate: so we always display
//		(otherwise, if we reload the page and the last gamestate is non-refreshing
//		we would see nothing). the_gs is null on reload, that's how we detect the
//		first message.
//	  - For certain callbacks that need the current gamestate, even if there was
//		no new display to update the callback itself. E.g., the proposing and voting
//		callbacks that need to know how many people have proposed or voted so far,
//		but won't see the update because they are non-refreshing.
var the_gs = null;

var diffusion_params = {
		host   : 'klander-l102aw.eu.diffusion.cloud',
		port   : 443,
		secure : true,
		principal : 'admin',
		credentials : 'admin'
	}
var TopicSpecification = diffusion.topics.TopicSpecification;
var TopicType = diffusion.topics.TopicType;


var add_player_topic_callback = function(path, newValue, player_names) {
	if (debug) console.log('Got string update for topic: ' + path, newValue);
	var player_name = newValue;
	if (player_names.indexOf(player_name) == -1) {
		player_names.push(player_name);
		if (debug) console.log("Added player " + player_name + ", players are " + player_names);
	} else {
		if (debug) console.log("Player " + player_name + "repeated, players are " + player_names);
	}
	$(".manage p").text("Jugadores: " + player_names);
}


var start_button_click_callback = function(session, player_names) {
	if (player_names.length <2) {
		alert("Cannot play with less than 2 players.");
		return;
	}
	the_gs = new GameState(108, player_names)
	chief_start_game(session);
	execute_command(session, the_gs, {"name": "init"});
	setCookie("game_is_on", true, 3600 * 6);  // game on for 6 hours
}


var send_player_name = function(session, player_name) {
	if (debug) console.log("Sending player name " + player_name);
	setCookie("player_name", player_name, 3600 * 6);
	session.topics.updateValue('dixit/player_names',
							   player_name,
							   diffusion.datatypes.string());
	if (!is_chief) subscribe_to_gamestate(session, player_name);
	$("div.waiting_for_session").show();
	$("div.player_name").hide();
}


var setup_player_name = function(session) {
	player_name = getCookie("player_name");
	$("div.game_container").hide();
	$("div.manage_container").show();
	if (!player_name) {
		if (debug) console.log("No player name registered yet.");
		$("div.waiting_for_session").hide();
		$("div.player_name").show();
		$(".player_name_form").on("submit", function(event){
			event.preventDefault();
			if($(".player_name_form input").val()) send_player_name(session, $(".player_name_form input").val());
		});
	} else {
		send_player_name(session, player_name);
	}
}


var setup_game = function(session) {
	var player_names = [];
	return session.topics.remove('dixit/gamestate')
	.then(() => {
		console.log('Removed topic dixit/gamestate');
		}, reason => {
		console.log('Failed to remove topic dixit/gamestate: ', reason);
	})
	.then(() => {
		return session.topics.add('dixit/player_names',
								  new TopicSpecification(TopicType.STRING, {DONT_RETAIN_VALUE: "true"}))})
	.then(result => {
		console.log('Added topic: ' + result.topic);
		session
			.addStream('dixit/player_names', diffusion.datatypes.string())
			.on('value', function(path, specification, newValue, oldValue) {
				add_player_topic_callback(path, newValue, player_names);
			});
		// Subscribe to the topic. The value stream will now start to receive notifications.
		session.select('dixit/player_names');
		}, reason => {
			console.log('Failed to add topic: ', reason);
		})
	.then(() => {
		$(".manage button").click(() => {start_button_click_callback(session, player_names);});
		return session;
	});
}

var subscribe_to_gamestate = function(session, player_name) {
	session
		.addStream('dixit/gamestate', diffusion.datatypes.json())
		.on('value', function(path, specification, newValue, oldValue) {
			var is_first = (the_gs === null);
			gs = JSON.parse(newValue.get());
			var refresh = gs.refresh;
			if (debug) console.log('Got JSON update for topic: ' + path, gs);
			if (is_first) {
				the_gs = new GameState(gs.n_cards, gs.player_names);
			}
			the_gs = Object.assign(the_gs, gs);
			if (refresh || is_first) display(session, gs, player_name);
			if (is_first && is_chief) {
				if (debug) console.log("Subscribing to dixit/command after receiving dixit/gamestate, the_gs is ", the_gs);
				subscribe_to_command(session, the_gs);
			}
		});
	// Subscribe to the topic. The value stream will now start to receive notifications.
	session.select('dixit/gamestate');
}

var subscribe_to_command = function(session, gs) {
	if (debug) console.log("Subscribing to dixit/command, gs is ", gs);
	session
		.addStream('dixit/command', diffusion.datatypes.json())
		.on('value', function(path, specification, newValue, oldValue) {
			if (debug) console.log('Got JSON update for topic: ' + path, newValue.get());
			execute_command(session, gs, JSON.parse(newValue.get()));
		});
	// Subscribe to the topic. The value stream will now start to receive notifications.
	session.select('dixit/command');
}


var chief_start_game = function(session) {
	session.topics.add('dixit/gamestate',
					   new TopicSpecification(TopicType.JSON)).then(function(result) {
        console.log('Added topic: ' + result.topic);
		subscribe_to_gamestate(session, getCookie("player_name"));
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});

    session.topics.add('dixit/command',
					   new TopicSpecification(TopicType.JSON,
											  {DONT_RETAIN_VALUE: "true"})).then(function(result) {
        console.log('Added topic: ' + result.topic);
		// When called from the start button, the chief has the_gs already
		// and can and should start to listen to commands.
		// (The alternative is starting from reload, we will get
		// the gs on the first message after subscribing to dixit/game_state)
		if (the_gs) {
			if (debug) console.log("Subscribing to dixit/command " +
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
		session.topics.updateValue('dixit/command',
								   JSON.stringify(obj),
								   diffusion.datatypes.json());
	}
}

var execute_command = function(session, gs, obj) {
	if (debug) {
		console.log("Executing command, name=" + obj["name"] + ", arg1=" + obj["arg_1"], ", arg2=" + obj["arg_2"]);
		console.log("gs is ", gs);
		console.log("the_gs is ", the_gs);
	}
	var refresh = false;
	if (obj["name"] == "init") {
		refresh = gs.init();
	}
	if (obj["name"] == "propose") {
		refresh = gs.propose(obj["arg_1"], obj["arg_2"], false);
	}
	if (obj["name"] == "sing") {
		refresh = gs.sing(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "vote") {
		refresh = gs.vote(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "next_round") {
		refresh = gs.next_round();
	}
	gs.refresh = refresh;
	session.topics.updateValue('dixit/gamestate',
							   JSON.stringify(gs),
							   diffusion.datatypes.json());
}


var set_hand_size = function(gs, player_idx) {
	var window_width = $(window).width();
	var card_height = $("div.hand").height();
	var card_width = card_height * 11 / 16;
	card_width = Math.min(
		card_width,
		(window_width - card_width) / (gs.player_hands[player_idx].length - 1));
	for (var i=0; i<gs.player_hands[player_idx].length; i++) {
		$(".hand img").eq(i).show()
			.attr("src", "img/Img_"+(gs.player_hands[player_idx][i]+1)+".jpg")
			.css("left", i * card_width)
			// .css("width", (100 / gs.cards_per_player - 2) + "%")
			.css("background-color", "");
		if (gs.player_hands[player_idx][i] == gs.candidates[player_idx]) {
			$(".hand img").eq(i).css("background-color", "red");
		}
	}
}

var set_candidates_size = function(gs, player_idx) {
	var window_width = $(window).width();
	var card_height = $("div.candidates").height();
	var card_width = card_height * 11 / 16;
	card_width = Math.min(
		card_width,
		(window_width - card_width) / (gs.n_players - 1));
	for (var i=0; i<gs.candidates.length; i++) {
		$(".candidates img").eq(i).show()
			.attr("src", "img/Img_"+(gs.candidates_shown[i]+1)+".jpg")
			.css("left", i * card_width)
			.css("background-color", "");
		if (gs.votes[player_idx] == gs.candidates_shown[i]) {
			$(".candidates img").eq(i).css("background-color", "red");
		}
		$(".candidates p.proposed").eq(i).css("left", i * card_width).outerWidth(card_width, true);
	}
}

var display_voted = function(gs, voted_whom) {
	var window_width = $(window).width();
	var card_height = $("div.candidates").height();
	var card_width = card_height * 11 / 16;
	card_width = Math.min(
		card_width,
		(window_width - card_width) / (gs.n_players - 1));
	var cum_votes = [];
	for (var i = 0; i < gs.candidates_shown.length; i++)
		cum_votes.push(0);
	var cum_votes_so_far = cum_votes.slice();
	for (var i=0; i<gs.n_players; i++) {
		if (voted_whom[i] == -1) continue;
		cum_votes[voted_whom[i]]++;
	}
	console.log("Cum votes:" + cum_votes + ", voted_whom:" + voted_whom);
	for (var i=0; i<gs.n_players; i++) {
		idx = voted_whom[i];
		if (idx == -1) continue;
		if (gs.candidates_shown[idx] == gs.candidates[gs.turn]) color="blue";
		else color="red";
		$(".candidates p.voted").eq(i).show()
			.text(gs.player_names[i])
			.css("color", color)
			.css("left", idx * card_width)
			.css("top", 0.3 * card_height + 0.7 * card_height * cum_votes_so_far[idx] / cum_votes[idx])
			.outerWidth(card_width, true);
		cum_votes_so_far[idx]++;
	}	
	for (var i=gs.n_players; i<$(".candidates p.voted").length; i++) {
		$(".candidates p.voted").eq(i).hide().text("");
	}
}



var display = function(session, gs, player_name) {
	$("div.manage_container").hide();
	$("div.game_container").show();
	$("div.zoom").hide();
	$(".zoom p").stop().off('click').hide();
	$(".zoom img").hide().attr("src", "").off('click');
	$(window).off("resize");
	var player_idx = gs.player_names.indexOf(player_name);
	var mano_name = gs.player_names[gs.turn];
	if (player_idx < 0) {
		console.error("Couldn't find player name " + player_name + " among names " + gs.player_names);
	}
	if (debug) {
		console.log("Displaying, player hands: ");
		console.log(gs.player_hands);
		console.log("Deck index: " + gs.deck_index);
		console.log("Candidates: " + gs.candidates);
	}
	
	// Cards in hand
	set_hand_size(gs, player_idx);
	$(window).resize(() => set_hand_size(gs, player_idx));
	
	for (i=gs.player_hands[player_idx].length; i<$(".hand img").length; i++) {
		$(".hand img").eq(i).hide().attr("src", "");
	}
	$(".hand img").off('click');
	if(gs.stage <= 1) {
		$(".hand img").on('click',function(){
			var idx_in_hand = $(".hand img").index($(this));
			var card_id = gs.player_hands[player_idx][idx_in_hand];
			var ptext = $(".zoom p");
			var propose_this = function() {
				$(".hand img").css("background-color", "");
				$(".hand img").eq(idx_in_hand).css("background-color", "red");
				$(".zoom img").hide();
				ptext.stop().hide();
				var candidates_so_far = 0;
				for (var i=0; i<gs.n_players; i++) {
					// use the_gs so it is up-to-date even without refresh
					if (the_gs.candidates[i] != -1) candidates_so_far ++;
				}
				console.log("Candidates so far: " + candidates_so_far);
				pub_command(session, {
					name: "propose",
					arg_1: player_idx,
					arg_2: card_id,
				});
				if (candidates_so_far < gs.n_players - 1)
					$(".candidate_text p").show().text("Gracias. Puedes cambiar si quieres");
			}
			$(".zoom img").attr("src", "img/Img_"+(card_id+1)+".jpg")
				.off("click")
				.show();
			ptext.off("click").stop();
			if ((gs.stage==0 && gs.turn==player_idx) || (gs.stage==1 && gs.turn!=player_idx)) {
				$(".zoom img").on("click", propose_this);
				ptext.text("Elegir esta").on("click", propose_this).show();
				function startAnimation() {
					ptext.animate({opacity: 0}, 3000);
					ptext.animate({opacity: 1}, 3000, startAnimation);
				}
				startAnimation();
			} else {
				$(".zoom img").on("click", () => {$(".zoom img").hide()});
			}
			$("div.zoom").css("top", $("div.candidate_text").offset().top)
				.height("50%")
				.show();
		})
	}
	
	// Song
	if (gs.stage == 0 && player_idx == gs.turn) {
		if (gs.song == "") {
			$("p.song_caption").text(player_name + ", elige carta y di algo");
		} else {
			$(".song_form input").val(gs.song);
			$("p.song_caption").text("Ahora elige carta");
		}
		$("p.song_display").hide();
		var send_song = function() {
			pub_command(session, {
				name: "sing",
				arg_1: player_idx,
				arg_2: $(".song_form input").val()
			});
			$("p.song_caption").text("Ahora elige carta");
		}
		$(".song_form").show().on("submit", function(event){
			event.preventDefault();
			send_song();
		});
		$(".song_form input.the_song").on("focusout", function(event){
			send_song();
		});
	} else if(gs.stage == 1) {  // show proposal
		$("p.song_caption").text(mano_name + " ha dicho: ");
		$("p.song_display").show().text(gs.song).textfill({ maxFontPixels: 48 });
		$(".song_form input.the_song").off("focusout");
		$(".song_form").off("submit").hide();
	} else if(gs.stage >= 2) {  // show vote
		$("p.song_caption").text(mano_name + " dijo: ");
		$("p.song_display").show().text(gs.song).textfill({ maxFontPixels: 48 });
		$(".song_form input.the_song").off("focusout");
		$(".song_form").off("submit").hide();
	} else if(gs.stage == 0) {  // non-mano
		$("p.song_caption").text("Esperando a que " + mano_name + " cante");
		$("p.song_display").hide();
		$(".song_form input.the_song").off("focusout");
		$(".song_form").off("submit").hide();
	}
	
	// Candidate player names
	if (gs.stage < 3) {
		$(".candidates p").hide().text("");
	} else {
		for (var i=0; i<gs.candidates_shown.length; i++) {
			var idx = gs.candidates.indexOf(gs.candidates_shown[i]);
			var bg_color = (idx == gs.turn) ? "yellow":"white";
			$(".candidates p.proposed").eq(i).show()
				.css("background-color", bg_color)
				.text(gs.player_names[idx]);
		}
		for (var i=gs.candidates.length; i<$(".candidates p.proposed").length; i++) {
			$(".candidates p.proposed").eq(i).hide().text("");
		}
		var voted_whom = [];
		for (var i=0; i < gs.n_players; i++) {
			if (i != gs.turn) voted_whom.push(gs.candidates_shown.indexOf(gs.votes[i]));
			else voted_whom.push(-1);
		}
		display_voted(gs, voted_whom);
		$(window).resize(() => display_voted(gs, voted_whom));
	}

	// Candidates
	if (gs.stage <= 1) {
		$(".candidates img").hide().attr("src", "");
	}
	if (gs.stage == 0) {
		$(".candidate_text p").hide();
	} else if (gs.stage == 1) {
		if (player_idx == gs.turn) {
			$(".candidate_text p").show().html("Esperando a que los dem&aacute;s elijan sus cartas");
		} else if (gs.candidates[player_idx] == -1) {
			$(".candidate_text p").show().text("Elige carta, " + player_name);
		} else {
			$(".candidate_text p").show().text("Gracias. Puedes cambiar si quieres");
		}
	} else if (gs.stage >= 2) {
		set_candidates_size(gs, player_idx);
		$(window).resize(() => set_candidates_size(gs, player_idx));
		for (var i=gs.candidates.length; i<$(".candidates img").length; i++) {
			$(".candidates img").eq(i).hide().attr("src", "");
		}	
		$(".candidates img").off('click');
		if (gs.stage == 2) {
			if (player_idx == gs.turn) {
				$(".candidate_text p").show().html("Aqu&iacute las propuestas, " +
												   "esperando votos");
			} else if (gs.votes[player_idx] == -1) {
				$(".candidate_text p").show().html("Aqu&iacute las propuestas, a votar");
			} else {
				$(".candidate_text p").show().html("Gracias. Puedes cambiar si quieres");			}
			$(".candidates img").on('click',function(){
				var idx_in_proposals = $(".candidates img").index($(this));
				var card_id = gs.candidates_shown[idx_in_proposals];
				if (card_id == gs.candidates[player_idx]) return;

				$(".zoom img").attr("src", "img/Img_"+(card_id+1)+".jpg").show();
				var ptext = $(".zoom p");
				if (player_idx != gs.turn) {
					var vote_this = function() {
						$(".candidates img").css("background-color", "");
						$(".candidates img").eq(idx_in_proposals)
							.css("background-color", "red");
						ptext.stop().hide();
						$("div.zoom").hide();
						var votes_so_far = 0;
						for (var i=0; i<gs.n_players; i++) {
							// use the_gs so it is up-to-date even without refresh
							if (the_gs.votes[i] != -1) votes_so_far ++;
						}
						console.log("Votes so far: " + votes_so_far, " the_gs.votes: " + the_gs.votes);
						pub_command(session, {
							name: "vote",
							arg_1: player_idx,
							arg_2: card_id,
						});
						if (votes_so_far < gs.n_players - 2)
							$(".candidate_text p").show()
								.text("Gracias. Puedes cambiar si quieres");
					}
					ptext.text("Votar esta")
						.off("click")
						.on("click", vote_this)
						.show();
					$(".zoom img").off("click").on("click", vote_this);
					function startAnimation() {
						ptext.fadeToggle(1000, "swing", startAnimation);
					}
					startAnimation();
				} else {  // let the mano just see the zoomed card
					$(".zoom img").off("click").on("click", () => {$(".zoom img").hide()});
					ptext.off("click").hide();
				}
				$("div.zoom").css("top", "3%").height("50%").show();
			});
		} else {  // stage 4
			$(".candidate_text p").show().text("Resultados:");
		}
	}
	
	// Scoreboard
	for (i=0; i<gs.n_players; i++) {
		$(".player_names td").eq(i).text(gs.player_names[i]);
		$(".player_scores td").eq(i).text(gs.scores[i]);
	}
	
	// Next round
	if (gs.stage < 3 || !is_chief) {
		$(".next_round button").hide();
	} else {
		$(".next_round button").show()
		.on("click", function() {
			$(".next_round button").off("click").hide();
			pub_command(session, {"name": "next_round"});
		});
	}

}

var session_promise = diffusion.connect(diffusion_params);
session_promise.catch(error => {alert("Error setting up Diffusion session:" + error)});

if (is_chief) {
	if (!getCookie("game_is_on")) {
		$("div.game_container").hide();
		$("div.manage_container").show();
		if (debug) { console.log("Setting up game.");}
		session_promise.then(setup_game).then(setup_player_name);
	} else {
		console.log("game_is_on value " + getCookie("game_is_on"));		
		if (debug) { console.log("Game is on.");}
		session_promise.then(chief_start_game);
	}
} else {
	$("div.manage").hide();
	if (debug) { console.log("Connecting to game.");}
	session_promise.then(setup_player_name);
}


	