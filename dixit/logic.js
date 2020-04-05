var debug = true;
var is_chief = window.location.search.substring(1).indexOf("chief") >= 0;
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
	setCookie("player_name", player_name);
	session.topics.updateValue('dixit/player_names',
							   player_name,
							   diffusion.datatypes.string());
	if (!is_chief) subscribe_to_gamestate(session, player_name, false);
	$("div.waiting_for_session").show();
	$("div.player_name").hide();
}


var setup_player_name = function(session) {
	player_name = getCookie("player_name");
	if (!player_name) {
		if (debug) console.log("No player name registered yet.");
		$("div.waiting_for_session").hide();
		$("div.player_name").show();
		$(".player_name input").on("keypress", function(e) {
			if(e.which == 13 && $( this ).val()) send_player_name(session, $( this ).val());
		});
		$(".player_name button").on("click", function() {
			if($(".player_name input").val()) send_player_name(session, $(".player_name input").val());
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
			gs = JSON.parse(newValue.get());
			if (debug) console.log('Got JSON update for topic: ' + path, gs);
			display(session, gs, player_name);
			if (is_chief & !the_gs) {
				the_gs = new GameState(gs.n_cards, gs.player_names);
				the_gs = Object.assign(the_gs, gs);
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
	$("div.manage").hide();
	$("div.waiting_for_session").hide();
	$("div.player_name").hide();

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
		if (the_gs) {
			if (debug) console.log("Subscribing to dixit/command on start game, the_gs is " + the_gs);
			subscribe_to_command(session, the_gs);
		}
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});
}

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
	if (obj["name"] == "init") {
		gs.init();
	}
	if (obj["name"] == "propose") {
		gs.propose(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "sing") {
		gs.sing(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "vote") {
		gs.vote(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "next_round") {
		gs.next_round();
	}
	session.topics.updateValue('dixit/gamestate',
							   JSON.stringify(gs),
							   diffusion.datatypes.json());
}


var bigger_size = function() {
	if (typeof $(this).data('origwidth')=='undefined') {
		$(this).data('origwidth',$(this).width());
		$(this).data('origmarginleft',$(this).css("marginLeft"));
		$(this).data('origmarginright',$(this).css("marginRight"));
		$(this).data('origmargintop',$(this).css("marginTop"));
		$(this).data('origmarginbottom',$(this).css("marginBottom"));
		$(this).data('origzindex',$(this).css("zIndex"));
		$(this).data('origposition',$(this).css("position"));
	}
	$(this).stop().css("position", "absolute").animate({
		width:"50%",
		marginLeft:"-12.5%", marginRight:"-12.5%",
		marginTop:"0%", marginBottom:"-35%",
		zIndex: 2,
	});
}


var restore_size = function() {
	$(this).stop().css("position", $(this).data('origposition')).animate({
		width:$(this).data('origwidth'),
		marginLeft:$(this).data('origmarginleft'),
		marginRight:$(this).data('origmarginright'),
		marginTop:$(this).data('origmargintop'),
		marginBottom:$(this).data('origmarginbottom'),
		zIndex:$(this).data('origzindex'),
	});
}


var display = function(session, gs, player_name) {
	$("div.manage").hide();
	$("div.waiting_for_session").hide();
	$("div.player_name").hide();
	player_idx = gs.player_names.indexOf(player_name);
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
	for (i=0; i<gs.player_hands[player_idx].length; i++) {
		$(".hand img").eq(i).show()
			.attr("src", "img/Img_"+(gs.player_hands[player_idx][i]+1)+".jpg")
			.css("width", (100 / gs.cards_per_player - 2) + "%");
	}
	console.log((100 / gs.cards_per_player - 2) + "%");
	for (i=gs.player_hands[player_idx].length; i<$(".hand img").length; i++) {
		$(".hand img").eq(i).hide().attr("src", "");
	}	
	$(".hand img").off('mouseenter').off('mouseLeave').off('click');
	if((gs.stage==0 && gs.turn==player_idx) || (gs.stage==2 && gs.turn!=player_idx)) {
		$(".hand img")
			.on('mouseenter', bigger_size)
			.on('mouseleave',restore_size)
			.on('click',function(){
			pub_command(session, {
				name: "propose",
				arg_1: player_idx,
				arg_2: $(".hand img").index($(this)),
			})
		});
	}
	
	// Song
	if (gs.stage == 1) {
		if (player_idx == gs.turn) {
			$(".song img").show().attr("src", "img/Img_"+(gs.candidates[player_idx]+1)+".jpg");
			$(".song p").text("Ahora di algo:");
			$(".song input").show().on("keypress",function(e) {
				if(e.which == 13) {
					pub_command(session, {
						name: "sing",
						arg_1: player_idx,
						arg_2: $(".song input").val()
					});
				}
			})
		} else {
			$(".song img").attr("src", "");
			$(".song p").text("Esperando a que " + gs.player_names[gs.turn] + " diga algo");
			$(".song input").hide();
		}
	} else if (gs.stage == 0) {
		if (player_idx == gs.turn) {
			$(".song p").text("Es tu turno, " + gs.player_names[player_idx] + ", elige carta.");
		} else {
			$(".song p").text("Esperando a que " + gs.player_names[gs.turn] + " elija carta.");
		}
		$(".song img").hide().attr("src", "");
		$(".song input").hide();
	} else {
		$(".song img").hide().attr("src", "");
		$(".song p").text(gs.player_names[gs.turn] + " ha dicho: " + gs.song);
		$(".song input").hide();
	}
	
	// Candidate player names
	if (gs.stage < 4) {
		$(".candidate_players p").hide().text("");
	} else {
		for (i=0; i<gs.candidates.length; i++) {
			$(".candidate_players p").eq(i).show()
				.text(gs.player_names[gs.candidates.indexOf(gs.candidates_shown[i])])
				.css("width", (100 / gs.cards_per_player - 2) + "%");
		}

		for (i=gs.candidates.length; i<$(".candidate_players p").length; i++) {
			$(".candidate_players p").eq(i).hide().text("");
		}
	}

	// Candidates
	if (gs.stage <= 2) {
		$(".candidates img").hide().attr("src", "");
	}
	if (gs.stage <= 1) {
		$(".candidate_text p").hide();
	} else if (gs.stage == 2) {
		if (player_idx == gs.turn) {
			$(".candidate_text p").show().html("Esperando a que los dem&aacute;s elijan sus cartas.");
		} else {
			$(".candidate_text p").show().text("Elige carta, " + gs.player_names[player_idx] + ".");			
		}
	} else if (gs.stage >= 3) {
		for (i=0; i<gs.candidates.length; i++) {
			$(".candidates img").eq(i).show()
				.attr("src", "img/Img_"+(gs.candidates[i]+1)+".jpg")
				.css("width", (100 / gs.cards_per_player - 2) + "%");
		}
		for (i=gs.candidates.length; i<$(".candidates img").length; i++) {
			$(".candidates img").eq(i).hide().attr("src", "");
		}	
		$(".candidates img").off('mouseenter').off('mouseLeave').off('click');
		if (gs.stage == 3 && player_idx == gs.turn) {
			$(".candidate_text p").show().html("Estas son las propuestas, esperando a los votos.");
		} else if (gs.stage == 3) {
			$(".candidate_text p").show().text("Estas son las propuestas, hora de votar.");
			$(".candidates img")
				.on('mouseenter', bigger_size)
				.on('mouseleave', restore_size)
				.on('click',function(){
				pub_command(session, {
					name: "vote",
					arg_1: player_idx,
					arg_2: gs.candidates[$(".candidates img").index($(this))],
				})
			});
		} else {  // stage 4
			$(".candidate_text p").show().text("Y estos son los resultados.");
		}
	}
	
	// Scoreboard
	for (i=0; i<gs.n_players; i++) {
		$(".player_names td").eq(i).text(gs.player_names[i]);
		$(".player_scores td").eq(i).text(gs.scores[i]);
	}
	
	// Next round
	if (gs.stage < 4 || !is_chief) {
		$(".next_round button").hide();
	} else {
		$(".next_round button").show()
		.on("click", function() {
			pub_command(session, {"name": "next_round"});
		});
	}

}

var session_promise = diffusion.connect(diffusion_params);
session_promise.catch(error => {alert("Error setting up Diffusion session:" + error)});

if (is_chief) {
	if (!getCookie("game_is_on")) {
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


	
