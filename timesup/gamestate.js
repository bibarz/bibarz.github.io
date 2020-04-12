const allEqual = arr => arr.every( v => v === arr[0] );

var shuffle = function (array) {

	var currentIndex = array.length;
	var temporaryValue, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {
		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		temporaryValue = array[currentIndex];
		array[currentIndex] = array[randomIndex];
		array[randomIndex] = temporaryValue;
	}

	return array;

};

var GameState = function(cards) {
	// this.n_players = player_names.length;
	// this.player_names = player_names.slice();
	// this.cards_per_player = 6;
	this.round_time = 60;
	this.deck = cards.slice();
	this.stage = 0;  // 0 - Describir 1 - ONe word 2 - Gesto
	// this.turn = 0;
	this.deck_index = 0;
	this.player_id = -1;
	this.player_start_time = 0;
	this.guessed_last_round = -1;
    // refresh is not really a state but a message field. When we do
	// a state update that does not require refreshing (e.g., a proposal),
	// we still may need to publish it, it is not enough that the chief keeps
	// track of the change, because if the chief reloads, it will get the
	// state from the topic and forget about the non-refresh state change.
	// So we always publish the new state, but the topic listener checks
	// the refresh field to decide whether to update the display or not.
	this.refresh = true;
	
	this.init = function() {
		if (debug) {
			console.log("Initializing game state");
		}
		shuffle(this.deck);
		this.deck_index = 0;
		this.stage = 0;
		this.player_id = -1;
		this.player_start_time = 0;
		this.guessed_last_round = -1;
		return true;  // always refresh
	}
	
	this.set_round_time = function(time) {
		this.round_time = time;
		return false;
	}

	this.new_round = function(player_id, player_time) {
		this.player_id = player_id;
		this.player_start_time = player_time;
		return true;
	}

	this.end_round = function(cards_guess, cards_pass, cards_delete) {
		this.player_id = -1;
		this.player_start_time = 0;
		this.guessed_last_round = cards_guess.length;
		for (var i=0; i<cards_delete.length; i++) {
			idx = this.deck.indexOf(cards_delete[i]);
			if (idx == -1) {
				console.error("Got card " + cards_delete[i] + " to delete but is not in deck!")
			} else {
				console.log("Deleting card " + cards_delete[i] + " from deck")
				this.deck[idx] = "";			
			}
		}
		for (var i=0; i<cards_pass.length; i++) {
			idx = this.deck.indexOf(cards_pass[i]);
			if (idx == -1) {
				console.error("Got card " + cards_pass[i] + " to pass but is not in deck!")
			} else {
				console.log("Passing on card " + cards_pass[i])
				this.deck.push(cards_pass[i])
				this.deck[idx] = "";			
			}
		}
		this.deck = this.deck.filter(x => x.length > 0);
		var new_deck_index = this.deck_index + cards_guess.length;
		if (new_deck_index > this.deck.length) {
			console.error("Deck index was " + this.deck_index + " and got "
						  + cards_guess.length + " guessed, going out of bounds!")
		}
		if (new_deck_index >= this.deck.length) {
			this._next_stage();
		} else {
			this.deck_index = new_deck_index;
		}
		return true;
	}

	this._next_stage = function() {
		this.stage++;
		if (debug) {
			console.log("Next stage, " + this.stage);
		}
		shuffle(this.deck);
		this.deck_index = 0;
	}
}
