// HTML/属性操作系メソッド
//$('h1').text("Hello world :)");
$('#title').append("<br>はじめまして JQuery");

// HTML/属性操作系メソッド
//$("#myphoto img").attr("src", "myphoto2.jpg");


//CSS系メソッド
$('body').css("background-color","yellow");

$('#myphoto img').css({
    "border":"white solid 8px",
    "width":"400px"
});


//エフェクト系メソッド
$('.target').text("bye-bye").fadeOut("slow");

// イベント系メソッドとイベントハンドラ

$("button:eq(0)").click(function(){
    alert('やれば、できるじゃないか');
});


//- 問題1
// DOM（タグ）の操作

$("button:eq(1)").click(function(){
    $(this).clone(true).insertAfter(this);
});



//- 問題2
//　動きをつける

$("button:eq(2)").click(function(){
       $(this).animate({
           marginLeft: "300px"
       },3000).animate({
           marginTop: "200px"
       },1000).animate({
           marginLeft: "0px"
       },3000).animate({
           marginTop: "0px"
       },1000);
});


//- 問題3

var num = 1;
$("button:eq(3)").click(function(){
    num = (num == 4) ? 1 : (++num);
    console.log(num);
    // JSの三項演算子
    // 変数 = 条件式 ? trueの時の値 : falseの時の値 ;
    $("#myphoto img").attr("src", "myphoto"+num+".jpg");
});


//- 問題4
// 関数と連動

function str_count(str){
    return   str.length; //文字の長さを知る
}


$("button:eq(4)").click(function(){
    var answer = $('#my_text').val(); //text boxの値を持ってくる
    var result = str_count(answer); //関数が実行されて、戻り値GET
    alert(result+"文字いれたね？");
});


