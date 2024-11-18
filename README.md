# video_processor

<p align="center">
  <img alt="original" src="assets/original.gif" width="100%">
</p>

<p align="center">
  Original Video
</p>

<p align="center">
  <img alt="eye" src="assets/eye.gif" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="nose" src="assets/nose.gif" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="mouth" src="assets/mouth.gif" width="30%">
</p>

.container {
  display: flex;
}
.container.space-around {
  justify-content: space-around;
}
.container.space-between {  
  justify-content: space-between;
}

<p>Using <code>justify-content: space-around</code>:</p>
<div class="container space-around">
  <div>A</div>
  <div>B</div>
  <div>C</div>
</div>

<hr />

<p>Using <code>justify-content: space-between</code>:</p>
<div class="container space-between">
  <div>A</div>
  <div>B</div>
  <div>C</div>
</div>