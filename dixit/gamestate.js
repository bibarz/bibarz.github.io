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
	this.stage = 0;  // 0 - Mano picks card; 1 - Mano sends text; 2 - Players pick cards; 3 - Players vote; 4 - Results
	this.turn = 0;
	this.deck_index = 0;
	this.song = "";
	this.player_hands = [];
	this.candidates = [];
	this.candidates_shown = [];
	this.votes = [];
	this.scores = [];
	
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
	}
	
	this.propose = function(player_idx, card_idx) {
		if(debug) {
			console.log("Proposing, player_idx=" + player_idx + ", card_idx=" + card_idx);
			console.log("Player_hands are:");
			console.log(this.player_hands);

		}
		if(card_idx < 0 || card_idx >= this.player_hands[player_idx].length) {
			throw ("Card index " + card_idx + " out of bounds for player " +
				   player_idx + " with " + this.player_hands[player_idx].length +
				   " cards in his hand.");
		}
		if(this.stage == 0) {  // propuesta del mano
			if (player_idx != this.turn) {
				throw ("In stage 0, proposal from player " + player_idx + " while mano is "+ this.turn);
			}
			if(this.candidates[player_idx] != -1) {
				throw ("The mano seems to have proposed twice! Candidate idx was " + this.candidates[player_idx]);
			}
			this.candidates[player_idx] = this.player_hands[player_idx][card_idx];
			this.player_hands[player_idx].splice(card_idx, 1);
			this.stage = 1;
		} else if (this.stage == 2) {  // propuestas de los jugadores
			if (player_idx == this.turn) {
				throw ("In stage 2, proposal from mano (player " + player_idx + ")");
			}
			if(this.candidates[player_idx] == -1) {
				this.candidates[player_idx] = this.player_hands[player_idx][card_idx];
				this.player_hands[player_idx].splice(card_idx, 1);
			} else {  // player has changed his mind
				var tmp = this.candidates[player_idx]
				this.candidates[player_idx] = this.player_hands[player_idx][card_idx];
				this.player_hands[player_idx][card_idx] = tmp;
			}			
			if (this.candidates.every( v => v != -1 )) {  // everyone has proposed
				if(debug) console.log("All players have proposed");
				this.candidates_shown = shuffle(this.candidates.slice());
				this.stage = 3;
			}
		} else {
			throw "Got a proposal in stage " + this.stage;
		}	
	}
	
	this.sing = function(player_idx, song) {
		if(debug) {
			console.log("Player " + player_idx + " sings " + song);
		}
		if(this.stage == 1) {  // propuesta del mano
			if (player_idx != this.turn) {
				throw ("In stage 1, song from player " + player_idx + " while mano is "+ this.turn);
			}
			this.song = song;
			this.stage = 2;
		} else {
			throw "Got a song in stage " + this.stage;
		}
	}
	
	this.vote = function(player_idx, card_id) {
		if(this.stage != 3) { // wrong stage
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
		if (n_votes != this.n_players - 1) return;
		
		// Everyone (save the mano) has voted
		if (not_voted_idx != this.turn) {
			throw ("Looks like the mano (player " + this.turn + " is not the " +
				   "one player that didn't vote, the votes are " + this.votes);
		}
		// Update scores
		var mano_votes = 0;
		for (var i = 0; i < this.n_players; i++) {
			if (i == this.turn) continue;
			candidate_idx = this.candidates.indexOf(this.votes[i])
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
		if (mano_votes > 0 && mano_votes < this.n_players) {
			this.scores[this.turn] += 3;  // someone but not everyone guessed, mano wins
		} else {  // all or none guessed, everyone except mano gets 2 points
			for (var i = 0; i < this.n_players; i++) {
				if (i == this.turn) continue;
				this.scores[i] += 2;
			}
		}		
		this.stage = 4;  // show results
	}
}
