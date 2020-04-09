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

var GameState = function(n_cards, player_names) {
	this.n_players = player_names.length;
	this.player_names = player_names.slice();
	this.cards_per_player = 6;
	this.n_cards = n_cards;
	this.deck_order = [...Array(n_cards).keys()];
	this.stage = 0;  // 0 - Mano picks card and text; 1 - Players pick cards; 2 - Votes; 3 - Results
	this.turn = 0;
	this.deck_index = 0;
	this.song = "";
	this.player_hands = [];
	this.candidates = [];
	this.candidates_shown = [];
	this.votes = [];
	this.scores = [];
    // refresh is not really a state but a message field. When we do
	// a state update that does not require refreshing (e.g., a proposal),
	// we still must publish it, it is not enough that the chief keeps
	// track of the change, because if the chief reloads, it will get the
	// state from the topic and forget about the non-refresh state change.
	// So we always publish the new state, but the topic listener checks
	// the refresh field to decide whether to update the display or not.
	// (It is slightly tricky because even if refresh=False we need to
	// display when it is the first message after a reload).
	this.refresh = true;
	
	this.init = function() {
		if (debug) {
			console.log("Initializing game state");
		}
		shuffle(this.deck_order);
		this.deck_index = 0;

		this.stage = 0;
		this.turn = 0;
		this.song = "";

		// Deal cards
		this.player_hands.length = 0;
		for (var i=0; i<this.n_players; i++) {
			var player_hand = [];
			for (var j=0; j<this.cards_per_player; j++) {
				player_hand.push(this.deck_order[this.deck_index++]);
			}
			this.player_hands.push(player_hand);
		}
		if (debug) {
			console.log("Initial player hands: ")
			console.log(this.player_hands);
		}

		// Clear state
		this.candidates.length = 0;
		this.candidates_shown.length = 0;
		this.votes.length = 0;
		this.scores.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.candidates.push(-1);
			this.votes.push(-1);
			this.scores.push(0);
		}
		return true;  // always refresh
	}
	
	this.next_round = function() {
		this.stage = 0;
		this.song = "";
		this.turn = (this.turn + 1) % this.n_players;
		if (this.deck_index <= this.n_cards - this.n_players) {
			// Deal one card per player
			for (var i=0; i<this.n_players; i++) {
				this.player_hands[i].push(this.deck_order[this.deck_index++]);
			}
		}
		// Clear state
		this.candidates.length = 0;
		this.candidates_shown.length = 0;
		this.votes.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.candidates.push(-1);
			this.votes.push(-1);
		}
		return true;  // always refresh
	}
	
	this.propose = function(player_idx, card_id) {
		if(debug) {
			console.log("Proposing, player_idx=" + player_idx + ", card_id=" + card_id);
			console.log("Player_hands are:");
			console.log(this.player_hands);
		}
		if(this.player_hands[player_idx].indexOf(card_id) == -1) {
			throw ("Card id " + card_id + " not in player " +
				   player_idx + "'s hand, " + this.player_hands[player_idx]);
		}
		if (this.stage > 1) {
			throw "Got a proposal in stage " + this.stage;
		}
		if(this.stage == 0 && player_idx != this.turn) {
			throw ("In stage 0, proposal from player " + player_idx +
				   " while mano is "+ this.turn);
		}
		if(this.stage == 1 && player_idx == this.turn) {
				throw ("In stage 1, proposal from mano (player " + player_idx + ")");
		}

		this.candidates[player_idx] = card_id;

		if (this.stage == 0 && this.song) {
			this.stage = 1;
			return true;
		}
		if (this.stage == 1 &&
			this.candidates.every( v => v != -1 )) {  // everyone has proposed
			if(debug) console.log("All players have proposed");
			for (var i=0; i < this.n_players; i++) {
				var card_idx = this.player_hands[i].indexOf(this.candidates[i]);
				if (card_idx == -1) throw (
					"After all players proposed (candidates " + this.candidates +
					"), found that card id " + card_id + " is not in player " +
					i + "'s hand, " + this.player_hands[i]);
				this.player_hands[i].splice(card_idx, 1);
			}
			this.candidates_shown = shuffle(this.candidates.slice());
			this.stage = 2;
			return true;
		}
		return false;
	}
	
	this.sing = function(player_idx, song) {
		if(debug) {
			console.log("Player " + player_idx + " sings " + song);
		}
		if(this.stage == 0) {  // propuesta del mano
			if (player_idx != this.turn) {
				throw ("Song from player " + player_idx + " while mano is "+ this.turn);
			}
			this.song = song;
			if(this.candidates[player_idx] != -1) {
				this.stage = 1;
				return true;
			} else {
				return false;
			}
		}
		throw "Got a song in stage " + this.stage;
	}

	this.vote = function(player_idx, card_id) {
		if(this.stage != 2) { // wrong stage
			throw "Got a vote in stage " + this.stage;
		}
		if (player_idx == this.turn) {
			throw ("Got a vote from the mano player, " + player_idx);
		}
		if(card_id < 0 || card_id >= this.n_cards) {
			throw ("Card id " + card_id + " out of bounds for deck of " +
				   this.n_cards + " cards")
		}
		var idx = this.candidates.indexOf(card_id);
		if(idx == -1) {
			throw ("Player " + player_idx + " voted for card id " + card_id +
				   " but available cards are " + this.candidates);
		}
		this.votes[player_idx] = card_id;
		var n_votes = 0;
		var not_voted_idx = -1;
		for (var i = 0; i < this.n_players; i++) {
			if (this.votes[i] >=0) {
				n_votes ++;
			} else {
				not_voted_idx = i;
			}
		}
		if (n_votes == this.n_players) {
			throw "Apparently all players (even the mano) have voted! Votes are " + this.votes;
		}
		if (n_votes != this.n_players - 1) return false;
		
		// Everyone (save the mano) has voted
		if (not_voted_idx != this.turn) {
			throw ("Looks like the mano (player " + this.turn + " is not the " +
				   "one player that didn't vote, the votes are " + this.votes);
		}
		// Update scores
		var mano_votes = 0;
		for (var i = 0; i < this.n_players; i++) {
			if (i == this.turn) continue;
			var candidate_idx = this.candidates.indexOf(this.votes[i]);
			if (candidate_idx == -1) {
				throw ("Vote for non-existing candidate, candidates are " + this.candidates + 
					   ", votes are " + this.votes);
			}
			if (candidate_idx == this.turn) {
				this.scores[i] += 3;  // player guessed the mano's card
				mano_votes += 1;
			} else {
				this.scores[candidate_idx] += 1;  // voted for candidate_idx, candidate gets a point
			}
		}
		if (mano_votes > 0 && mano_votes < this.n_players - 1) {
			this.scores[this.turn] += 3;  // someone but not everyone guessed, mano wins
		} else {  // all or none guessed, everyone except mano gets 2 points
			for (var i = 0; i < this.n_players; i++) {
				if (i == this.turn) continue;
				this.scores[i] += 2;
				var candidate_idx = this.candidates.indexOf(this.votes[i]);
				if (candidate_idx == this.turn) {
					this.scores[i] -= 3;  // undo the guess score, everyone guessed
				}
			}
		}		
		this.stage = 3;  // show results
		return true;
	}
}
