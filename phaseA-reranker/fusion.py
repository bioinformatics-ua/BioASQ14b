from ranx import Run, fuse
import click


@click.command()
@click.argument("runs", nargs=-1, type=click.Path())
@click.option("--out")
@click.option("--method", default="rrf")
def main(runs, out, method):
    ranx_runs = [Run.from_file(run) for run in runs]
    fuse(ranx_runs, method=method).save(out)


if __name__ == "__main__":
    main()
