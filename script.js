import {weight1, weight2} from "./weights.js"

const model = tf.sequential({
	layers: [
		tf.layers.dense({
			inputShape: [64],
			activation: "relu",
			units: 32
		}),
		tf.layers.dense({
			units: 10,
			activation: "softmax"
		})
	]
})

model.compile({optimizer: tf.train.adam(0.001), loss: tf.losses.meanSquaredError})

//model.summary()

model.layers[0].setWeights([tf.tensor2d(weight1, [64, 32]), tf.zeros([32])])
model.layers[1].setWeights([tf.tensor2d(weight2, [32, 10]), tf.zeros([10])])

function draw(){
	let x = [], y = [], u = "", m = [], res;
	let num = Math.pow(10, 6), x_max = -num, x_min = num, y_max = -num, y_min = num;


	svg.ontouchstart = () => {
			u = "";
			m.push([]);
			clearTimeout(res);
			inp1.style.display = "none";
		    inp2.style.display = "none";
	}

	svg.ontouchmove = e => {
		e.preventDefault()
		x.push(e.touches[0].clientX);
		y.push(e.touches[0].clientY);
		if(x[x.length - 1] > x_max){x_max = x[x.length - 1]};
		if(x[x.length - 1] < x_min){x_min = x[x.length - 1]};
		if(y[y.length - 1] > y_max){y_max = y[y.length - 1]};
		if(y[y.length - 1] < y_min){y_min = y[y.length - 1]};
		u += x[x.length - 1] + " " + y[y.length - 1] + " ";
		m[m.length - 1] = "M" + u;
		path.setAttribute("d", m)
	}

	svg.ontouchend = () => {
		res = setTimeout(() => {
				if((x_max - x_min) < 20){x_max += 50; x_min -= 50}
				if((y_max - y_min) < 20){y_max += 50; y_min -= 50}
				field(x, y, x_max, x_min, y_max, y_min);
				m = [];
				u = "";
				x = [];
				y = [];
				x_max = -num; x_min = num; y_max = -num; y_min = num;
			}, 1500)
	}

	
}

function field(x, y, x_max, x_min, y_max, y_min){
	let field = [], out;
	let x_correct, y_correct;
	for(let i = 0; i < 8; i++){
		field[i] = [];
		for(let j = 0; j < 8; j++){
			field[i][j] = 0;
		}
	}

	for(let i = 0; i < x.length; i++){
		x_correct = Math.trunc(8 * (x[i] - x_min) / (x_max - x_min));
		y_correct = Math.trunc(8 * (y[i] - y_min) / (y_max - y_min));
		if(x_correct == 8){x_correct--}
		if(y_correct == 8){y_correct--}
		field[y_correct][x_correct] = 1;
	}
	field = [].concat(...field);
	out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
	if(!confirm(print(field))){
		let res = prompt("set a number")
		out[res] = 1;
		train_model(field, out);
		inp1.style.display = "block";
		inp2.style.display = "block";
		inp1.setAttribute("value", model.layers[0].getWeights()[0].dataSync());
		inp2.setAttribute("value", model.layers[1].getWeights()[0].dataSync());
	}
}


function train_model(field, out){
	let x = tf.tensor2d(field, [1, 64]);
	let y = tf.tensor2d(out, [1, 10]);
	model.fit(x, y, {epochs: 10}).then(() => {
		if(!confirm(print(field))){
			return train_model(field, out);
		}
	})
}

function print(field){
	let res = model.predict(tf.tensor2d(field, [1, 64])).dataSync();
	let max = Math.max.apply(null, res);
	//let out = [];
	//res.filter(item => {out.push(item.toFixed(3))})
	return res.indexOf(max);
}

draw();