/******************
Code by Vamoss
Original code link:
https://openprocessing.org/sketch/1799766

Author links:
http://vamoss.com.br
http://twitter.com/vamoss
http://github.com/vamoss
******************/
// https://github.com/amandaghassaei/gpu-io/blob/main/examples/fluid/index.js
var fluid;

// https://github.com/amandaghassaei/canvas-capture
var canvasCapture;
var capturedFrames = 0;

// Touch events.
const activeTouches = {};
const TOUCH_DIAMETER = 25;

function setup() {	
	//hack to enable p5js WEBGL2
	if(GPUIO.isWebGL2Supported()){
		p5.RendererGL.prototype._initContext = function() {
			try {
				this.drawingContext =
					this.canvas.getContext('webgl2', this._pInst._glAttributes) ||
					this.canvas.getContext('experimental-webgl', this._pInst._glAttributes);
				if (this.drawingContext === null) {
					throw new Error('Error creating webgl2 context');
				} else {
					const gl = this.drawingContext;
					gl.enable(gl.DEPTH_TEST);
					gl.depthFunc(gl.LEQUAL);
					gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
					this._viewport = this.drawingContext.getParameter(
						this.drawingContext.VIEWPORT
					);
				}
			} catch (er) {
				throw er;
			}
		};
	}
	
	var renderer = createCanvas(windowWidth, windowHeight, WEBGL);
	
	canvasCapture = window.CanvasCapture.CanvasCapture;
	const RECORD_FPS = 60;

	// Init a simple gui.
	const gui = new dat.GUI();
	
	fluid = new Fluid(renderer.canvas, gui);
	fluid.resize(width, height);
	
	canvasCapture.dispose();
	canvasCapture.init(renderer.canvas, { showRecDot: true, showDialogs: true, showAlerts: true, recDotCSS: { left: '0', right: 'auto' } });
	canvasCapture.bindKeyToVideoRecord('v', {
		format: CanvasCapture.WEBM,
		name: 'screen_recording',
		fps: RECORD_FPS,
		quality: 1,
	});
}

function draw() {
	fluid.draw();
	
	// Screen recording.
	canvasCapture.checkHotkeys();
	if (canvasCapture.isRecording()) {
		canvasCapture.recordFrame();
		capturedFrames++;
		console.log(`Recording duration: ${(capturedFrames / RECORD_FPS).toFixed(2)} sec`);
	} else {
		capturedFrames = 0;
	}
}

function pointerMoved(e){
	const current = [e.clientX, e.clientY];
	if (activeTouches[e.pointerId] === undefined) {
		activeTouches[e.pointerId] = {
			current: current,
		}
		return;
	}
	var last = activeTouches[e.pointerId].last = activeTouches[e.pointerId].current;
	activeTouches[e.pointerId].current = current;

	if (current[0] == last[0] && current[1] == last[1]) {
		return;
	}

	fluid.pointerMove(current, last);
}

function mouseReleased(e){
	delete activeTouches[e.pointerId];
}

function keyPressed(){
	if(key == "p"){
		fluid.savePNG();
	}
}

function windowResized() {
  fluid.resize(windowWidth, windowHeight);
}